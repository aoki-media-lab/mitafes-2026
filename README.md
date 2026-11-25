# 統合版の来場者カウンター

手元の精度改善版 `accurate_counter` に、[Mac版](https://github.com/Koki0724/mitafes/tree/84f080f4dbb01c9247ea6dd7bfe8492a5af5aa7f) の運用面の工夫を取り込んだ、録画向けの統合版です。

ガラス側は全画面と通過帯の2入力、姿勢と胴体による追跡、遮蔽後に現れる人物の通過救済を維持しています。Mac版は元の追跡実装と共通で、新しい精度改善を含む版ではありません。そのため、追跡をMac版の古い方式へ戻す変更はしていません。

## 取り込んだ改善

- 動画が途中で破損・終了した場合に、最後まで処理できたと誤報しない。
- 読み込み側のキューが満杯でも、停止要求で終了できる。例外時に動画を閉じる。
- 再開時に動画のパス・順序・サイズ・更新日時、モデルのSHA-256、設定、実装、依存バージョンを照合する。
- SQLiteに保存済みの位置より先の未確定ログを、再開前に切り戻す。
- 予定した観測数と実際の処理数を照合する。短い動画を長時間試験として扱わない。
- 時間帯ごとの遅延を記録し、等速入力時の遅延増加を確認できる。
- Apple Silicon/MPSとCUDAを選べる。MPSの実演算確認、Macのメモリ単位への対応を取り込む。
- 重みをチェックサムで検証して取得する。既存モデルを上書きしない。

加えて、2入力の通過を照合する処理をSQLiteの一時索引へ変更しました。映像全体の軌跡をRAMへ読み込まず、必要な時刻の軌跡を最大128件までキャッシュします。同じ22分49秒のログで、最大RSSが107.06 MiBから52.30 MiBへ減り、通過イベントと軌跡の出力は完全一致しました。索引は照合終了時に削除します。

## Linux / CUDAで実行

Python 3.11を使います。GPU・ドライバーに対応するCUDA版PyTorchを先に導入し、依存ライブラリとモデルを準備してください。検証環境はQuadro P6000 / PyTorch 2.10.0+cu126です。

```sh
git clone --branch develop-gpu https://github.com/Koki0724/mitafes.git
cd mitafes
python3.11 -m venv .venv
source .venv/bin/activate
# ここでGPUに対応するCUDA版PyTorch 2.10.0 / torchvision 0.25.0を導入
python -m pip install -r requirements.txt
python scripts/download_models.py
python doctor.py --platform cuda

# ガラス側の精度改善プリセット。最初の1分を検証
python run_accuracy.py \
  --platform cuda --gpus 0 1 \
  --models-dir ./models --source /path/to/glass.mp4 \
  --max-seconds 60 --output runs/my_glass_60s
```

`--max-seconds` を外すと末尾まで処理します。GPUが1枚なら `--gpus 0` を指定します。ガラス側のこのプリセットは、元の設置位置の2880×1620 / 25fps用です。

見下ろし側は従来の追跡・通過判定を使います。

```sh
python run.py \
  --platform cuda --devices 0 1 --batch 8 --camera overhead \
  --source /path/to/overhead.mp4 \
  --start-seconds 670 --max-seconds 60 \
  --record-tracks --output runs/my_overhead_60s
```

670秒は元の基準区間の開始位置です。別動画の先頭なら `--start-seconds 0` にします。

## ガラス側を1パスで高速に処理（`run_fused.py`）

`run_accuracy.py` と同じ2入力（全画面・x=40〜80%の切り抜き）、同じモデル・入力形状・閾値・追跡・照合で、録画を1回だけデコードします。傘モデルは両入力で同一の全画面推論だったため1回にまとめ、3つのネットワークを別々のCUDAストリームで重ねて実行します。CUDAのFP32のみで、`--resume` には未対応です。

```sh
python run_fused.py --source /path/to/glass.mp4 --output runs/my_glass_fused
```

結果は `run_accuracy.py` と同じく `full/`・`roi/`・`result/` に出力します。22分49秒の録画（RTX 3080 Ti、250W制限）で686.7秒→403秒になり、照合後の通過603件の向き・フレーム・時刻が完全に一致しました。詳細は [VALIDATION.md](VALIDATION.md) を参照してください。

## カメラからのリアルタイム処理（`live_fused.py`）

RTSPカメラ（またはNVR）の映像を、`run_fused.py` と同じ推論・追跡に流し、2入力の照合も `stream_merge.py` で逐次に行います。照合の条件は `merge_indexed.py` と同じで、判定に必要な前後のフレームがそろう約7秒後に roi 側の通過を確定します。保持するのは直近約18秒分だけです。

```sh
# URLは ~/.mitafes_rtsp（chmod 600）から読みます。ログやoutputには認証情報を出しません。
echo 'rtsp://USER:PASS@NVR_IP:554/live/1/1/avm' > ~/.mitafes_rtsp && chmod 600 ~/.mitafes_rtsp
python live_fused.py --duration 600 --output runs/live_10min

# 録画を25fpsの実時間で流す仮想カメラ
python live_fused.py --file /path/to/glass.mp4 --duration 1400 --output runs/live_virtual
```

### 両方向カウント（入場・退場・構内人数）

数える向きはセットアップJSONの `profile.count_directions` で決まります。2026-11-23からの本番設定 `reference/setups/glass_view3_both_20261123.json` は両方向を数え、`in_direction`（`rtl`＝入場）と `direction_labels`（`入場`/`退場`）を持ちます。Webビューは入場・退場・その差（6:00からこの門で入って、まだこの門から出ていない人数）と、時間帯別の出入り（入場は上向き、退場は下向き）・差の推移を表示します。退場を数えていない時間がその日にあると、差は表示しません。

右→左だけを残していた日の退場は、保存済みの軌跡から推論なしで数え直せます（右→左が実運用と完全一致することを `--verify` で確認）:

```sh
python scripts/recount_tracks.py --run runs/live/20261122_060000 --output runs/recount/20261122_both --verify
```

結果を `runs/live/<日付>_recount_*/`（`live_events.jsonl` に `clock`、`window.json` に `start`/`end`/`directions`）に置くと、Webビューの日別表示がその区間を置き換えます（11/22は `20261122_recount_both`）。

### 影・夕方向けの調整（`glass_view3_tuned_20261123.json`）

11/22 のNVR録画（9:55〜11:05、12:55〜14:05、16:55〜18:05）で調整した設定です。

- `detect_conf: 0.03`：検出の下限。影の中の黒っぽい服の人や、夕方にブレた人は信頼度0.05〜0.28で検出されるため、これを残して追跡をつなぎます。
- `pose_new_track_thresh: 0.1`：新規追跡の閾値（従来は full 0.2 / roi 0.3）。
- `rule_min_confidence: 0.3`：補う規則（late_birth / stitch / late_death）は、最高信頼度が0.3に達した追跡だけが使えます。薄い断片が数に化けるのを防ぎます。

入場（右→左）は 3,395→3,511（朝）、3,812→4,053（昼）、842→982（夕）。`schedule`（`[{"from":"HH:MM","profile":{...}}]`）で時間帯ごとに値を変えられます（`live_fused.py` が10秒ごとに確認し、実行中に切り替えます）。

閾値の比較は、`run_fused.py --dump-detections --conf 0.03` で検出を一度保存し、`replay_fused.py --min-conf/--override` で推論なしに行います。違いは `scripts/diff_review.py`（片方だけが数えた通過の前後を切り出し）で目視確認します。

TP-Link VIGI NVRのライブ映像は `rtsp://IP:554/live/<チャンネル>/<1=メイン|2=サブ>/avm` です。入力は2880×1620 / 25fpsのみ受け付けます。処理が約2.6秒分以上遅れると、遅延が際限なく伸びないよう8観測単位で破棄し `dropped_frames` に数えます。切断時は再接続し、時間軸を経過時間ぶん進めます。10秒ごとの受信fps・遅延・破棄数・GPU電力を `live_status.jsonl` に記録し、終了時に逐次照合とオフライン照合の通過数を比較します。

## Apple Siliconで実行

MPSのコード経路を実装しましたが、**この統合版をMac実機で推論した速度・精度は未検証**です。以下は実機検証用の手順です。

```sh
# arm64 Python 3.11 / macOS 14以降
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.txt
python scripts/download_models.py
python doctor.py --platform mps

python run_accuracy.py --platform mps \
  --models-dir ./models --source /path/to/glass.mp4 \
  --max-seconds 60 --output runs/mac_glass_60s
```

MPSではFP32・バッチ1を使います。CUDA Graphは使いません。利用できるアクセラレーターがない場合、CPU推論への自動切り替えは行いません。環境変数でMPSのCPUフォールバックを有効にした場合は、純粋なGPU推論と測定条件が異なるため、`doctor.py` の出力も保存して比較します。

モデルは `--models-dir` で指定できます。省略時はこのフォルダーの `models/` を使い、この作業環境にある既存の `../cpu_counter/models/` も利用します。Linuxで新規環境を作る場合は、GPU・ドライバーに合うPyTorchを先に導入してください。ここではQuadro P6000と `torch 2.10.0+cu126` の組み合わせを使用しています。

## 分割録画と再開

```sh
# 欠落や重複のない連続した分割録画を順に指定
python run_accuracy.py --platform cuda --gpus 0 \
  --source part01.mp4 part02.mp4 part03.mp4 \
  --max-seconds 3600 --output runs/school_day

# 同じ入力・モデル・コード・設定で処理範囲を延長
python run_accuracy.py --platform cuda --gpus 0 \
  --source part01.mp4 part02.mp4 part03.mp4 \
  --max-seconds 7200 --output runs/school_day --resume
```

各ファイルで追跡をリセットせず、同じ寸法・fpsの分割録画を連続して処理します。コードや重みを変更した場合は新しい出力先を使います。従来版のチェックポイントをこの統合版へ持ち込む方式ではありません。

2入力の状態は `full/` と `roi/` にそれぞれ保存します。`result/events.csv` が照合後の通過一覧です。`pipeline_status.json` は実行中・失敗・完了の状態と、確認するログを示します。再開時の処理時間は今回追加した部分、通過数は累計です。

## 検証結果と範囲

[VALIDATION.md](VALIDATION.md) に数値と成果物を記載しています。前回の268件中266件、99.25%という検証サンプルの結果を維持しています。今回は新しい独立した映像での精度向上を実証した作業ではありません。

31件のテストでは、従来の2画角、画面下端の人物、遮蔽後の通過、破損・読み込み失敗、ログ切り戻し、再開条件、MPSの呼び出し、索引照合を確認します。

```sh
python -m unittest test_accuracy test_streaming test_integration test_reconciliation -v
```

録画の一括処理が対象です。8時間連続運転、実際の通信断を伴うRTSP運用、Macでの長時間の発熱・推論性能は別途検証が必要です。

このブランチの推奨入口は、ガラス側が `run_accuracy.py`、見下ろし側が `run.py --camera overhead` です。ガラス側を `run.py` 単体で実行しても、2入力の照合は行いません。追加のベンチマーク・頭部モデル・ONNX・TensorRT関連スクリプトは実験用で、上記の検証済み構成には使っていません。動画の描画、ffmpegデコーダー、NVENC出力には対応するffmpegが別途必要です。

動画、モデル重み、`runs/` の実行出力はGitに含めていません。回帰テスト用の検出データは `tests/fixtures/`、小さな検証記録は `validation/gpu_20261117/` に同梱しています。

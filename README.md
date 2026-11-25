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

TP-Link VIGI NVRのライブ映像は `rtsp://IP:554/live/<チャンネル>/<1=メイン|2=サブ>/avm` です。入力は2880×1620 / 25fpsのみ受け付けます。処理が約2.6秒分以上遅れると、遅延が際限なく伸びないよう8観測単位で破棄し `dropped_frames` に数えます。切断時は再接続し、時間軸を経過時間ぶん進めます。10秒ごとの受信fps・遅延・破棄数・GPU電力を `live_status.jsonl` に記録し、終了時に逐次照合とオフライン照合の通過数を比較します。

## 新しい画角：石畳の広場（`--view plaza`、2026-11-20〜）

カメラはVIGI C450（7C9E）で、石畳の広場を見下ろす画角に決まりました。NVRのアドレスは `192.168.222.2` です。人は右下と、左の通用口・左上の開口部との間を斜めに歩きます。計測線はその流れを横切る斜めの線（上端 x=0.549 → 下端 x=0.278）です。**rtl（画面の右→左、右下から左側へ）が最優先の方向**で、ltr（左→右）も同じ判定で数えます。

```sh
python live_fused.py --view plaza --duration 600 --output runs/plaza_live      # カメラから
python run_fused.py  --view plaza --source rec.mp4 --output runs/plaza_rec   # 録画から
```

画角ごとの設定は `views/*.json` にまとめました。内容は解像度、計測線（`line`）、対象範囲（`roi_polygon`）、2入力の切り抜き範囲です。`--view` を省略すると、従来のガラス側の設定（`views/glass.json`、縦線 x=0.6）で動作は変わりません。斜めの線の判定は `line_counter.py` にあり、縦線では従来の判定と完全に一致します（テストで確認）。

広場向けに追加・変更した点：

- 対象範囲から画面上端の帯を除外。上の通路に立ち止まる人や、そこを歩く人の脚を数えないためです。
- 線の近くで出現した追跡の救済（`birth_band`=画面幅の4%）。集団の中で隠れたまま線を越え、線の直後で初めて検出された人を、歩いた方向に数えます。生きている追跡に重なる重複枠は除外します。
- 途切れた追跡の引き継ぎ（`link_seconds`=0.6秒、`link_distance`=画面幅の8%）。画面の端で体の一部しか映らない人はIDが分裂しやすく、線を越えた後の断片から始まるため数え漏れていました。消えた直後の追跡と同じ場所に新しい追跡が現れた場合、確定済みの側を引き継ぎます。最後に反対側で見えていた追跡や、生きている追跡に重なる重複枠からは引き継ぎません。

夜間の対策（2026-11-23）：暗くなるとカメラは20fpsに落として露光を伸ばすため（11/21・22とも16:45〜06:10ごろ）、歩行者がブレ、黒い服の人や重なった2人の片方を見逃していました。計測の開始から16:00までは11/22までの設定のまま、`views/plaza.json` の `schedule` で **16:00からだけ検出の信頼度を0.1→0.05に下げます**。ライブは10秒ごとに時刻を確かめて切り替え（`detector_schedule` をログに出力）、録画は `run_fused.py --clock-start 'YYYY-MM-DD HH:MM:SS'` で同じ規則を当てます。予定に入れられるのはNMSで使う `conf` と `iou` だけです。検出の設定（`pose_model`・`conf`・`iou`・`tta_flip`・`enhance`）は `views.detector_settings` にまとめ、`run_fused.py --detector '{...}'` で上書きして比較できます。保存した検出からの再集計は `tools/replay_view.py --min-conf` で閾値も変えられます。夜の録画（20fps）をライブと同じ時間軸で数え直すには `run_fused.py --assume-fps 25`、その動画は `tools/render_counted.py --source-fps 20` です。反転TTA（`tta_flip`）と照合の `dedup_passages` は試しましたが、可視化動画の確認で二重カウントが目立ったため使っていません。結果は [VALIDATION.md](VALIDATION.md) を参照してください。

線の決め方と調整の手順：`tools/plot_paths.py` で歩行経路を重ねて描き、`run_fused.py --record-detections` で検出結果を保存します。`tools/replay_view.py --set '{...}'` を使うと、GPUを使わず数秒で設定を比較できます。差分は `tools/compare_events.py` で突き合わせ、`tools/review_sheets.py` で目視確認します。結果は [VALIDATION.md](VALIDATION.md) を参照してください。

## リアルタイム処理とWeb UI（`live_service.py` / `scripts/start_live.sh`）

Webページは常に動かしておき、カウントは毎日6:00〜19:00（ローカル時刻）だけ行います。計測時間外はカメラの映像を受信せず、GPUも使いません。

```sh
tmux new-session -d -s mitafes-live scripts/start_live.sh   # 開始（SSHを切っても動き続けます）
tmux attach -t mitafes-live                                 # ログを見る（離れるときは Ctrl-b → d）
tmux send-keys -t mitafes-live C-c                          # 停止（計測中なら集計を書き出してから終了）
HOURS=7-19 scripts/start_live.sh                            # 計測時間を変える場合
```

ブラウザで `http://<このPCのアドレス>:8080/` を開きます。Tailscale経由なら `http://100.121.149.12:8080/` です。

- 左：カメラ映像。カウント線、判定マージン、計測範囲（足元）、照合用切り抜き、人物、傘、追跡中の頭の位置と軌跡を重ねて表示します。右上の表示は「計測中」か「計測時間外」です。画質は なめらか（12.5コマ/秒・約8Mbps）、標準、省データ、非表示から選べます。処理は8枚ずつまとめて行うため、カメラから約1.2秒遅らせて等間隔で表示します。
- 右上：通過数。東門（広場）では **右→左＝入場、左→右＝退場** です（`views/plaza.json` の `entry_direction`。時間帯別に見ると右→左は朝、左→右は夕方に多いことから判断）。入場・退場と、その差の「今キャンパスにいる人数」（入場−退場、計測開始時を0人とする）を表示します。他の門の出入りは含まないため、他の門から入って東門から出る人が多い夕方にはマイナスになることがあります（11/22は終了時点で−759）。入場・退場は、計測中と、その日にデータがある間は当日の値を、それ以外は最後にデータがある日の値を表示します。「今キャンパスにいる人数」は計測中だけ表示します。`entry_direction` のない画角（ガラス側）は右→左・左→右の表示になり、人数の差は出しません。
- 右中：状態。経過時間、受信fps、遅延、破棄フレーム、再接続、GPU、表示fpsです。計測時間外は「-」になります。
- 右下：時間帯別の入場・退場の人数（2色の棒）と、キャンパスにいる人数の推移（1分ごとの累計の折れ線）。15分と1時間を切り替えられます。日付を選べ、「表で見る」で各時間帯の入場・退場と、時間帯の終わりの時点の在場人数を表で確認できます。

### カウントの可視化動画（`live_recorder.py`）

常駐サービスは、計測中のカウントを可視化動画として `~/Videos/mitafes-live/YYYY-MM-DD/HHMMSS.mp4` に保存します（1時間ごとに1ファイル）。保存しない場合は `--no-record-video`、場所を変える場合は `--record-video DIR` を付けます。`live_fused.py` 単体では `--record-video DIR` を付けたときだけ保存します。

- 灰色の枠：検出中で、まだ数えていない人。オレンジの太枠「右→左 #番号」：右→左でカウント済みの人。青の太枠「左→右 #番号」：左→右でカウント済みの人。数えた瞬間は通過地点に波紋と「+1」を出します。
- 左上：時刻、その日の方向別の人数（Web画面と同じ値）、直近のカウント。常駐サービスの既定は両方向（右→左＝オレンジ、左→右＝青）です。`live_fused.py` 単体の既定は右→左だけです（`--record-directions rtl ltr` で両方向）。
- 照合で追加されるカウントは約7秒後に確定するため、動画は8秒遅らせて書き出します（`dedup_passages` を使う場合は12秒）。動画の人数は集計と一致します。
- 1920×1080・12.5コマ/秒、H.264（NVENC）で、約2.7GB/時です。途中で停止・強制終了しても、それまでの分は再生できる形式（fragmented MP4）で書きます。空き容量が20GBを切ると録画だけを止めます。
- 書き出しが遅れても、カウント処理は待ちません（録画のコマを捨て、`live_status.jsonl` の `recorder.dropped_frames` に数えます）。録画が失敗してもカウントは続きます。

14時台（混雑）の録画を実時間で15分流した比較：遅延p50 0.49秒・p95 0.75秒で録画なしと同じ、破棄0、録画の欠落0。メモリは約1.1GB増えます（8秒分のコマを保持するため）。

録画済みの映像からは `tools/render_counted.py --source rec.mp4 --run runs/<run_fused.pyの出力> --clock-start 'YYYY-MM-DD HH:MM:SS'` で同じ表示の動画を作れます。

日ごとの合計と1分ごとの人数は `runs/live_daily/YYYY-MM-DD.json` に保存します。両方向の人数を記録し、画面にも表示します。同じ日のうちに再起動しても続きから数え、過去の日のファイルは残ります。処理の記録からの集計し直しは `tools/rebuild_daily.py runs/live_... [--overwrite]` で行えます。カメラ映像を出さない場合は `--no-snapshot` を付けます。外部への通信や外部ライブラリの読み込みはしません。

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

# develop-gpu の状態

録画向けのCUDA / MPS統合版。ガラス側は `run_accuracy.py`、見下ろし側は `run.py --camera overhead` が入口です。

- 元のMac版: `main`、取り込み元コミット `84f080f4dbb01c9247ea6dd7bfe8492a5af5aa7f`。
- 実際のCUDA再推論: ガラス側60秒62/62、見下ろし側60秒16/16が既存の基準軌跡と一致。
- 保存済み全編ログの照合結果: 606件、従来版とイベント・軌跡のハッシュが一致。
- 既存の採点サンプル: 266/268、余計なカウント0。全編や本番の99%以上を確認した結果ではありません。
- 1パス版 `run_fused.py`：22分49秒を686.7秒→403.1秒、照合後の通過603件が完全一致。RTSPライブ版 `live_fused.py`：10分間の実カメラ入力で破棄0、遅延p95 0.73秒。
- 新しい画角（広場、`--view plaza`）：斜めの計測線、上端の除外、途切れた追跡の引き継ぎ、線の近くで出現した追跡の救済。10分×4本で、差が出た通過はすべて目視確認済み（[VALIDATION](../VALIDATION.md)）。
- 常駐サービス `live_service.py`（`scripts/start_live.sh`）：Webページ（ポート8080）は常に動かし、カウントは毎日6:00〜19:00だけ行う。日ごとの集計を `runs/live_daily/` に保存する。
- 両方向カウント（2026-11-22）：Webに入場（右→左）・退場（左→右）と「今キャンパスにいる人数」（入場−退場、計測中のみ）を表示。
- 夜間の対策（2026-11-23）：広場は16:00から検出の信頼度を0.05に下げる（`schedule`）。それまでは11/22までの設定のまま。夜の3区間で761→827件（+9%）（[VALIDATION](../VALIDATION.md)）。
- Mac実機での推論と8時間の連続運転は未検証。
- push前に `python -m unittest discover -v` を実行し、57件が成功（2026-11-23、`test_detector.py` 3件・`test_stream_merge.py` 1件・`test_live_web.py` 2件を追加。以前は46件、2026-11-21）。記録は [tests.txt](../validation/gpu_20261117/tests.txt)。

セットアップは [README](../README.md)、検証範囲・測定値は [VALIDATION](../VALIDATION.md) を参照してください。動画・重み・実行ログは含めていません。

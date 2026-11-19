# Demo script (plan phase 6, T4)

A five-minute demonstration of the live system on the laptop: real D1 traffic goes in packet by
packet, decisions come out early, unknown apps are flagged, the budget knob moves the number of
packets read, and the drift monitor reacts to a late week. Everything runs locally from one
process: `python -m adl_etc.service` serves the API and the web UI.

Numbers to quote while presenting are in the "What to say" column and come from
`docs/results-phase-4.md`, `docs/results-phase-5.md` and the measurements in
`plans/phase-6-service-dashboard-demo.md`. Do not quote anything else.

## Rehearsal checklist (before the audience arrives)

1. **Model bundle present.** `results/models/production.json` exists. If not:
   `python scripts/make_production_bundle.py` (seconds; it checks every checkpoint hash).
2. **Shards complete.** `python scripts/check_shards.py --root results/kaggle-full/shards`.
3. **Port free.** On Windows a killed `uvicorn` can leave a process holding the port. The service
   checks the port at start and tells you if it is taken. To free it:
   `netstat -ano | findstr :8000`, then `taskkill /PID <pid> /F`. Or use `--override port=8001`.
4. **A fresh database per demo run** (the service creates it; the old file can be deleted):

   ```
   python -m adl_etc.service --override db_url=sqlite:///results/service/demo.db drift.window=1000 torch_threads=4
   ```

   `drift.window=1000` makes the drift monitor close a window every ~45 s instead of every ~4 min.
   Its validation reference was measured on 5,000-flow windows, so the spread is rescaled and the
   Drift page says alarms are approximate. Say so if asked.
5. **Wait for "Model 2fe7d91d2993 live"** in the page's top bar (`/health` says `ready`; about
   5 s). Open `http://127.0.0.1:8000/` in a browser at 100% zoom, window maximised.
6. **Pre-roll the drift week (about 5 minutes before starting).** In the console
   (`/#console/monitor`), replay **Week 2022 / 50**, 8,000 flows, 30 flows per second, speed-up 3.
   It takes about six minutes. The drift page then has a history to show. What to expect (measured
   in the 2026-11-18 rehearsal, after 4,000 week-31 flows and 6,000 week-50 flows): the week-31
   windows sit within about 1σ of validation; on week 50 next-packet surprise and the app-mix
   shift sit 1-3σ high and confidence 1-2σ low, and the app-mix CUSUM had reached 7.4 of its
   alarm level 8, so 8,000 flows should take it over. If no alarm shows, say what the bars show:
   the signals moved, the alarm needs them to stay moved.
7. When the pre-roll is done, open **Budget**, choose **Auto**, target **6**, Apply.

Start order: only the service. The UI is part of it; there is no second process.

## The script

| Time | Do | What to say |
|---|---|---|
| 0:00 | Story page, top. Let the packet animation run. | "Each line is an encrypted connection. Bars are packets: up goes to the server, down comes back. A line stops the moment the system decides: cyan means it named the app, rose means it called it unknown. It only ever sees sizes, timing and direction." |
| 0:30 | Scroll to **Three calls, made after every packet**; let the example play, then click **An app it never saw**. | "After every packet, three choices: flag as unknown, name it, or wait. Here a familiar app is named after six packets; an app held out of training is flagged at packet four." |
| 1:00 | **Open live console.** Monitor: replay **Week 2022 / 31**, 3,000 flows, 30 per second, speed-up 2, Start. | "This is a real week of CESNET traffic the model never trained on, replayed packet by packet through the same code the offline experiments used. The service's decisions match the offline replay exactly: 0 of 400 differed in the test." |
| 1:30 | Point at the tiles and the table. | "About six packets per flow on average. The True app column is green when the call was right. Rose rows are flows it refused to name; some of those apps were held out of training on purpose." |
| 2:00 | Click a **flagged unknown** row (or one named late). The Flow inspector opens. | "Here is one flow: its packets, the top three guesses after each packet, the confidence against the commit threshold, and the unknown score crossing its threshold at the moment it stopped." |
| 2:45 | **Budget**: target **4**, Apply. Wait about 30 s with the chart visible. | "Now I tell it to read fewer packets. Every 500 decisions a feedback loop moves the commit threshold. Watch the realised line follow the target down. Accuracy drops a little: offline that is about 81% at 4 packets against 88% at 6." |
| 3:30 | **Drift**. | "Before you arrived it replayed a December week, 24 weeks after the last training week. These signals need no labels. The app-mix shift and next-packet surprise moved away from validation; offline, the app-mix shift tracked the weekly F1 drop closely (Spearman −0.90) and first alarmed within a week of a 3-point drop." Point at the bars and the history chart. |
| 4:15 | **Experiments**, the criteria list at the top. | "Every hypothesis had a pass bar fixed before the runs. Some are met; some are not, for example beating P-ECHO, the confidence-threshold baseline. Both are reported." |
| 4:45 | Back to the story page, **What held up, and what did not**. | "One small model, 885,821 parameters, decides early, says unknown when unsure, and holds a budget. The full pass costs 2.49 ms per flow on this CPU; this live demo scores about 300 packets a second, recomputing each prefix." |

Reset between runs: stop the service (Ctrl+C), delete `results/service/demo.db*`, start again.

## Scene 2 (optional): PCAP upload

Monitor, **Upload a capture**: choose a `.pcap` (for example a USTC-TFC2016 file under
`data/raw/`), speed-up 5, Upload. Say: "The same pipeline reads a capture file. This one was
recorded somewhere else, on different networks and apps, so its predictions can be poor. That is
expected; the model was trained on CESNET's capture point." Handshake-only connections appear as
*no payload*: there is nothing to classify.

## Scene 3 (not built): the payment-API motivation

Roadmap DP-6a's second scene (`docs/future-work.md` FEP-1) needs a mock payment API and scripted
clients. It is not built. If asked: the project's results are about naming apps from encrypted
traffic; the fraud story is motivation only, and **the system does not detect fraud**.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Top bar says "Service offline" | The service is not running or is on another port. |
| "Loading model" does not clear | Look at the terminal: a missing file in the bundle is named there. Run `python scripts/make_production_bundle.py`. |
| Replay says "no shards for WEEK-…" | `data_root` in `configs/service.yaml` must point at `results/kaggle-full/shards`. |
| Decisions arrive slowly | Normal above ~25 new flows per second on this CPU: the replay slows itself to the engine (the queue stays at about 256 packets). Lower the rate. |
| Port held after a crash | `netstat -ano \| findstr :8000`, `taskkill /PID <pid> /F`. |

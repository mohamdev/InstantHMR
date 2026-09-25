# 06: Profile a 288 px input on the reference phones

**What to build:** decide whether the 288 px arm is deployable before GPU hours are spent on it. A random-weight B4 export at 288, profiled like ticket 03.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] NPU latency on Galaxy S22, S23, S24 and S25; CPU latency on S23, A73, A53, A14 and Pixel 8
- [ ] 100% NPU placement and fp16 parity on the S23
- [ ] Result recorded next to ticket 03's table, with the 224-vs-288 latency ratio per phone

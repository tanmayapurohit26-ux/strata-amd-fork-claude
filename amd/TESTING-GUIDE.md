# Strata on your RX 7900 XTX: testing guide

This is the AMD (ROCm) version of Strata, the engine that runs the 125B Qwen3.8-Flash-Next model with its experts split between the graphics card, RAM and CPU. It is set up for two model sizes:

- **IQ3_XXS**, which you already have
- **IQ3_S**, which you're downloading

## Where things stand

**Done and checked:**
- The whole engine and all of its test programs compile for your card (gfx1100).
- The small byte-level helpers I had to write for AMD were checked against NVIDIA's behaviour on 20 million random inputs, with zero mismatches.

**Not done yet:** nothing has run on a real AMD card. That's what this guide is for. I built everything in a cloud workspace with no graphics card, using an older ROCm (5.7) than yours (7.2.4). So expect a few rounds of "run it, send me the error, I fix it".

## What changed from the original

The original only works with NVIDIA's CUDA. I left its GPU code as it was and added a translation layer, so the same code compiles with AMD's HIP compiler. Where the two systems behave differently, I made targeted fixes:

- **Shared host memory.** The engine's GPU and CPU signal each other through memory both can see. On NVIDIA that memory is always kept in sync. On AMD it must be requested, so the AMD version asks for it explicitly. If this were wrong, the engine would freeze at the first token.
- **One attention scoring step.** On NVIDIA it uses special matrix hardware that your card doesn't have. On AMD it uses ordinary maths. The result is the same to within rounding, and I added a test for it.
- **A 64 KB limit.** One step needs about 80 KB of fast on-chip memory when checking 8 draft tokens at once. Your card allows 64 KB per group, so on AMD that case runs in two halves.
- **The installer (`setup.sh`).** It now recognises an AMD card and builds the engine with ROCm. It also builds the image encoder with ROCm, and reuses your existing second model file, explained next.
- **The monitor tab** on the web page now shows your AMD card's load, VRAM, temperature and power.

**Shared second file.** The second file of every model size (`...-00002-of-00002.gguf`, 28.8 GB) is the same file. It's the model's word lookup table, not the experts. For IQ3_S you only need shard 1. The installer links the existing shard 2 in under the IQ3_S name. The link points at the file you already have, so no disk space is used twice.

## Before you start

- **Stop your usual model server** (`serve-stop`) before any step below that runs the GPU. Both want nearly all the VRAM and RAM.
- **RAM is tight with IQ3_S.** It keeps about 50 GB of experts in RAM, so close Docker containers and browsers you don't need when running it. IQ3_XXS needs about 43 GB.
- **Port:** Strata defaults to port 8080, which your SearXNG already uses. The commands below use **8081**, the same port your `serve-*` scripts use, so your coding tools can reach it unchanged. Only one server can hold 8081 at a time.
- **Model folder:** I haven't seen your exact folder layout under `~/Local Models/ISTA-DASLab/`, so the commands below write it as `<IQ3_XXS folder>`. Replace that with the folder that holds `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`. This command lists what's there:

  ```
  ls -R ~/"Local Models"/ISTA-DASLab
  ```

## Step 1: put the code on the rig (one time)

Download `strata-amd-rocm.zip` from our chat onto the rig and put it in your home folder. This unpacks it into `~/Strata-amd`:

```
unzip ~/strata-amd-rocm.zip -d ~
```

This moves you into that folder. Every later command is run from there:

```
cd ~/Strata-amd
```

## Step 2: check the PC (one time)

This creates Strata's own private Python environment inside the folder and checks your hardware. It installs nothing system-wide and downloads nothing big.

```
./setup.sh --check
```

**What you should see:** a line like `GPU: AMD Radeon RX 7900 XTX (gfx1100), 24.0 GB VRAM, ROCm 7.2.4`, then your RAM and CPU. **AVX2** is expected for your CPU. Also a list saying which model sizes fit in your RAM.

**If it says "no NVIDIA GPU found ... and no AMD GPU either":** send me the output of this command. It shows what ROCm reports about your card:

```
/opt/rocm/bin/rocminfo
```

## Step 3: build and run the tests (one time; about 30 minutes)

This command:
1. Compiles the engine plus about 25 small test programs for your card.
2. Runs each test on the GPU.
3. Checks the IQ3_XXS model's real expert data three ways: on the GPU, on the CPU, and against llama.cpp's own reference maths.

It writes only inside the `Strata-amd` folder.

```
bash amd/check.sh "<IQ3_XXS folder>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
```

When your IQ3_S download has finished, run it once more with that file added at the end. The build is reused, so the second run is quick:

```
bash amd/check.sh "<IQ3_XXS folder>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" "<IQ3_S folder>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
```

**What you should see:** one `pass` or `FAIL` line per test, then a count.

**If anything fails:** send me the whole `amd/logs` folder. A failure isn't always a real problem. Some of the author's tests demand results identical to NVIDIA down to the last bit, and a few maths functions round slightly differently on AMD. I'll tell you which failures matter.

**Stop here if these fail:**
- **"BUILD FAILED"**: send `amd/logs/build.log`.
- **`device_selftest` or `native_qsa_score_test` fails**: those two check the basics of the port.

## Step 4: install IQ3_XXS (one time)

This command:
1. Prepares the model for Strata. It reads your existing files and doesn't copy them.
2. Downloads the MTP draft layer, about 5 GB from Hugging Face. MTP is the speed trick where the model guesses several tokens and checks them in one pass.
3. Compiles the engine, 10 to 20 minutes.
4. Compiles the image encoder with ROCm, which can take 30 to 60 minutes the first time.

`--no-start` stops it from starting the model at the end, so you can do that separately in step 6.

```
./setup.sh --setup --family qwen --model IQ3_XXS --gguf-dir "<IQ3_XXS folder>" --context 131072 --kv int8 --vision yes --experimental-speed-projection off --port 8081 --no-start
```

What the settings mean:
- `--context 131072`: 128K tokens of context, the largest the author recommends for IQ3 models with 64 GB of RAM.
- `--kv int8`: 8-bit storage for the conversation memory (the KV cache), the setting all the author's measurements used.
- `--vision yes`: the image encoder runs on the GPU.
- `--experimental-speed-projection off`: despite its name, this is an optional add-on that changes how the model answers (it makes it refuse less). It stays off.

## Step 5: install IQ3_S (one time, after the download finishes)

The same as step 4, with the IQ3_S folder. If shard 2 isn't in that folder, the installer links your IQ3_XXS copy in automatically and prints a line saying so. The engine and image encoder are already compiled, so this is much faster.

```
./setup.sh --setup --family qwen --model IQ3_S --gguf-dir "<IQ3_S folder>" --context 131072 --kv int8 --vision yes --experimental-speed-projection off --port 8081 --no-start
```

## Step 6: start the model and try it

You run this command **each session**. It asks which installed model to start (IQ3_XXS or IQ3_S), then loads it. Loading reads 43 to 50 GB into RAM and takes 30 to 90 seconds. Closing the terminal window stops the model.

```
./setup.sh
```

Once it's running, open `http://127.0.0.1:8081` for Strata's own chat page. The API for your coding tools is `http://127.0.0.1:8081/v1`.

A good first test is to send the same long prompt you used for your `serve-flash` measurements and compare speeds.

Should I add a shortcut such as `serve-strata` to your `llm-aliases.sh`? Then you wouldn't need to `cd` into the folder and type this each time.

## What to send me after each step

- **Step 2:** the printed output.
- **Step 3:** the summary lines, plus the `amd/logs` folder if anything failed.
- **Steps 4 and 5:** the last 30 or so lines if it stops with an error.
- **Step 6:**
  - Whether the first answer arrives.
  - The speed figures on the chat page.
  - The engine log `strata-iq3_xxs.log` (or `strata-iq3_s.log`) in the `Strata-amd` folder.

## If the model loads but never answers

This is the most likely failure. It would mean the GPU and CPU can't see each other's signals in shared memory. Stop it with Ctrl+C and start it again with one setting that forces all shared memory to stay in sync. This setting applies to this one run only:

```
HIP_HOST_COHERENT=1 ./setup.sh
```

If that fixes it, tell me. It means some memory allocation still needs the fix, and I'll find it.

## Things to know

- **Your coding harness.** Strata accepts thinking levels `none`, `low`, `medium` and `high` only. Oh My Pi sends `minimal` and will get an error. That's the same problem you patched in the llama.cpp template. I can make Strata accept `minimal` (treating it as `low`) once the basics work.
- **Access from your laptop.** Strata listens on the rig only by default. If you want your laptop to reach it as with your `serve-*` scripts, add `--host 0.0.0.0 --api-key <a long secret>` to the step 4 and 5 commands. With `0.0.0.0`, anyone on your network can use it without the key.
- **Speed.** The first goal is correct output. Speed tuning for AMD comes after. Parts of the engine were tuned for NVIDIA cards, and some AMD overheads, such as the cost of launching a batch of GPU work, are higher, so the first numbers may land below the author's.

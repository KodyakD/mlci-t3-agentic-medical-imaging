import os
import re
import subprocess
import sys
import time

KAGGLE = r"C:\Users\PC\AppData\Roaming\Python\Python313\Scripts\kaggle.EXE"
KERNEL = "abdelraoufgheyat/kermany-nih-merged"
ROOT = r"C:\Users\PC\Desktop\M2S1\MLCI\t3_agentic"
WEIGHTS = os.path.join(ROOT, "..", "weights")
IMAGES = os.path.join(ROOT, "..", "demo_images")


def fetch(dest, pattern, tries=12):
    os.makedirs(dest, exist_ok=True)
    for i in range(tries):
        r = subprocess.run(
            [KAGGLE, "kernels", "output", KERNEL, "-p", dest,
             "--file-pattern", pattern, "-q"],
            capture_output=True, text=True, timeout=3600)
        err = (r.stderr or "") + (r.stdout or "")
        if "429" in err or "Too Many Requests" in err:
            wait = min(60 * (i + 1), 600)
            print(f"[retry {i+1}] rate limited, sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.returncode != 0:
            print(f"[fail] rc={r.returncode}: {err[-1500:]}", flush=True)
        else:
            print(f"[ok] {pattern}", flush=True)
        return r.returncode == 0
    return False


if __name__ == "__main__":
    # 1) model weights first (everything depends on them)
    fetch(WEIGHTS, r".*densenet_merged\.keras$")
    print("weights present:",
          os.path.exists(os.path.join(WEIGHTS, "densenet_merged.keras")), flush=True)

    # 2) demo/test images from both domains (needed for eval + failure mining).
    #    Only the first images of each class: a full download of the 1500+ PNGs
    #    is what triggered the API rate limit in the first place.
    for domain, cls in (("clean_kermany", "NORMAL"), ("clean_kermany", "PNEUMONIA"),
                        ("clean_nih", "NORMAL"), ("clean_nih", "PNEUMONIA")):
        fetch(IMAGES, rf".*{domain}/test/{cls}/{cls}_[0-9]\.png$")

    n = sum(len(files) for _, _, files in os.walk(IMAGES))
    print("images downloaded:", n, flush=True)

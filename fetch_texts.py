"""Download the two public posts used as held-out prose (not redistributed here) and extract plain text.
  dl/blog.html.txt    <- https://rikiyat.github.io/blog/weight-space-programming/
  dl/report.html.txt  <- https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html
Also copies the blog text to bluebox/heldout_text.txt (the auditors' held-out text)."""
import html, os, re, shutil, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{HERE}/dl", exist_ok=True)
SRC = {"blog.html.txt": "https://rikiyat.github.io/blog/weight-space-programming/",
       "report.html.txt": "https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html"}
for name, url in SRC.items():
    t = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", t, flags=re.S); t = re.sub(r"<[^>]+>", " ", t)
    open(f"{HERE}/dl/{name}", "w").write(re.sub(r"\s+", " ", html.unescape(t)))
    print(name, os.path.getsize(f"{HERE}/dl/{name}"), "bytes")
shutil.copy(f"{HERE}/dl/blog.html.txt", f"{HERE}/bluebox/heldout_text.txt")

import sys, json, re, signal, os

# Support any working directory — resolve relative to this file's location
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

def _timeout(sig, frame):
    raise TimeoutError('hard timeout')
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(18)

url = sys.argv[1]
try:
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=8) as r:
        raw = r.read(80000).decode('utf-8', errors='ignore')
    for tag in ['script','style','nav','header','footer','aside']:
        raw = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', ' ', raw, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', raw)
    text = re.sub(r'\s+', ' ', text).strip()[:3500]
    doc_name = url.rstrip('/').split('/')[-1][:40] or url[:40]
    # Redirect all prints to stderr so stdout is clean JSON only
    import builtins
    _real_print = builtins.print
    def _print(*a, **kw):
        kw['file'] = sys.stderr
        _real_print(*a, **kw)
    builtins.print = _print
    from agents.ingestor import ingest_text
    doc = ingest_text(text, doc_name, max_chunks=2)
    doc['source_url'] = url
    builtins.print = _real_print  # restore
    sys.stdout.write(json.dumps(doc) + '\n')
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stdout.write(json.dumps({'error': str(e)}) + '\n')
    sys.exit(1)

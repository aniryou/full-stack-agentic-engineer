# Runs inside the sandbox: its only way out is $SANDBOX_PROXY_URL, and it never sees a key.
import http.client, json, os, socket

url = os.environ["SANDBOX_PROXY_URL"]
if url.startswith("unix:"):
    class Conn(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX)
            self.sock.connect(url[5:])
    c = Conn("egress-proxy")
else:
    host, _, port = url.split("//", 1)[1].partition(":")
    c = http.client.HTTPConnection(host, int(port or 80))
c.request("GET", "/api-stub/data")
r = c.getresponse()
data = json.loads(r.read())
print(r.status, data)
print("average high:", sum(d["high_c"] for d in data["forecast"]) / len(data["forecast"]))
print("any key in my environment:", any("sk-" in v for v in os.environ.values()))

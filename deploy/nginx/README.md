# Nginx TLS edge for the Model Gateway

Nginx terminates HTTPS and forwards requests only to the Gateway on
`127.0.0.1:8080`. Leaf model and pipeline ports remain loopback-only.

1. Replace `api.example.com` in `nginx.conf` with the production API domain.
2. Install the certificate and private key at the configured paths, or change
   the paths to those managed by the hosting platform/Certbot.
3. Start the Gateway with `GATEWAY_HOST=127.0.0.1` and
   `TRUST_PROXY_HEADERS=true`. Do not enable trusted proxy headers if clients
   can bypass Nginx and reach the Gateway directly.
4. Validate and reload Nginx:

```bash
sudo nginx -t -c /absolute/path/to/nginx.conf
sudo systemctl reload nginx
```

The public URL becomes `https://api.example.com/api/v1/...`. Port `8080` must
not be opened by the host firewall when Nginx is used on the same server.

The 28 MB edge body limit allows a 20 MB image encoded as Base64. Keep it in
sync with `MAX_REQUEST_MB`; tune rate and connection limits for real traffic.

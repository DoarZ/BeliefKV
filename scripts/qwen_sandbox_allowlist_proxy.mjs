#!/usr/bin/env node

import http from "node:http";
import net from "node:net";

const args = process.argv.slice(2);
const allowed = new Set();
for (let index = 0; index < args.length; index += 1) {
  if (args[index] !== "--allow" || index + 1 >= args.length) {
    console.error("usage: qwen_sandbox_allowlist_proxy.mjs --allow HOST:PORT [...]");
    process.exit(2);
  }
  const authority = new URL(`http://${args[index + 1]}`);
  const port = authority.port || "80";
  allowed.add(`${authority.hostname.toLowerCase()}:${port}`);
  index += 1;
}
if (allowed.size === 0) {
  console.error("at least one --allow HOST:PORT is required");
  process.exit(2);
}

function targetKey(hostname, port) {
  return `${hostname.toLowerCase()}:${String(port)}`;
}

function deny(response, target) {
  console.error(`[proxy] denied ${target}`);
  response.writeHead(403, { "content-type": "text/plain" });
  response.end("Forbidden\n");
}

const server = http.createServer((request, response) => {
  let target;
  try {
    target = new URL(request.url);
  } catch {
    deny(response, request.url || "invalid request target");
    return;
  }
  const port = target.port || (target.protocol === "http:" ? "80" : "443");
  if (target.protocol !== "http:" || !allowed.has(targetKey(target.hostname, port))) {
    deny(response, target.href);
    return;
  }

  const headers = { ...request.headers, host: target.host };
  delete headers["proxy-authorization"];
  delete headers["proxy-connection"];
  const upstream = http.request(
    {
      hostname: target.hostname,
      port,
      method: request.method,
      path: `${target.pathname}${target.search}`,
      headers,
    },
    (upstreamResponse) => {
      response.writeHead(
        upstreamResponse.statusCode || 502,
        upstreamResponse.headers,
      );
      upstreamResponse.pipe(response);
    },
  );
  upstream.on("error", (error) => {
    console.error(`[proxy] upstream error: ${error.message}`);
    if (!response.headersSent) {
      response.writeHead(502, { "content-type": "text/plain" });
    }
    response.end("Bad Gateway\n");
  });
  request.pipe(upstream);
});

server.on("connect", (request, clientSocket, head) => {
  let target;
  try {
    target = new URL(`http://${request.url}`);
  } catch {
    clientSocket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
    return;
  }
  const port = target.port || "443";
  if (!allowed.has(targetKey(target.hostname, port))) {
    console.error(`[proxy] denied CONNECT ${target.hostname}:${port}`);
    clientSocket.end("HTTP/1.1 403 Forbidden\r\n\r\n");
    return;
  }

  console.error(`[proxy] allowed CONNECT ${target.hostname}:${port}`);
  const upstream = net.connect(Number(port), target.hostname, () => {
    clientSocket.write("HTTP/1.1 200 Connection Established\r\n\r\n");
    upstream.write(head);
    upstream.pipe(clientSocket);
    clientSocket.pipe(upstream);
  });
  upstream.on("error", (error) => {
    console.error(`[proxy] CONNECT upstream error: ${error.message}`);
    clientSocket.end("HTTP/1.1 502 Bad Gateway\r\n\r\n");
  });
  clientSocket.on("error", () => upstream.destroy());
});

server.listen(8877, "::", () => {
  console.error(`[proxy] listening on :::8877; allowed=${[...allowed].join(",")}`);
});

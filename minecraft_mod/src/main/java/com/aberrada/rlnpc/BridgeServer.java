package com.aberrada.rlnpc;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

public class BridgeServer {
    private static final int PORT = 8765;

    private final EnvironmentManager environmentManager;
    private HttpServer server;

    public BridgeServer(EnvironmentManager environmentManager) {
        this.environmentManager = environmentManager;
    }

    public void start() {
        if (server != null) return;

        try {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", PORT), 0);

            server.createContext("/health", exchange -> sendJson(exchange, 200, "{\"status\":\"ok\"}"));

            server.createContext("/reset", exchange -> {
                if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                    sendJson(exchange, 405, "{\"error\":\"Use POST\"}");
                    return;
                }
                String body = readBody(exchange);
                String task = extractStringField(body, "task", "navigation");
                String result = environmentManager.reset(task);
                sendJson(exchange, 200, result);
            });

            server.createContext("/step", exchange -> {
                if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                    sendJson(exchange, 405, "{\"error\":\"Use POST\"}");
                    return;
                }
                String body = readBody(exchange);
                int action = extractAction(body);
                String result = environmentManager.step(action);
                sendJson(exchange, 200, result);
            });

            server.setExecutor(null);
            server.start();
        } catch (IOException e) {
            RLNpcMod.LOGGER.error("Failed to start bridge server", e);
        }
    }

    public void stop() {
        if (server != null) {
            server.stop(0);
            server = null;
        }
    }

    private void sendJson(HttpExchange exchange, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private String readBody(HttpExchange exchange) throws IOException {
        try (InputStream is = exchange.getRequestBody()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8).trim();
        }
    }

    private int extractAction(String body) {
        if (body == null || body.isBlank()) return 4;
        String trimmed = body.trim();
        try {
            return Integer.parseInt(trimmed);
        } catch (NumberFormatException ignored) {
        }

        int colon = trimmed.indexOf(':');
        if (colon >= 0) {
            String right = trimmed.substring(colon + 1).replaceAll("[^0-9\\-]", "");
            if (!right.isBlank()) {
                try {
                    return Integer.parseInt(right);
                } catch (NumberFormatException ignored) {
                }
            }
        }
        return 4;
    }

    private String extractStringField(String body, String field, String defaultValue) {
        if (body == null || body.isBlank()) return defaultValue;
        String pattern = "\"" + field + "\"";
        int start = body.indexOf(pattern);
        if (start < 0) return defaultValue;
        int colon = body.indexOf(':', start + pattern.length());
        if (colon < 0) return defaultValue;
        int q1 = body.indexOf('"', colon + 1);
        int q2 = body.indexOf('"', q1 + 1);
        if (q1 < 0 || q2 < 0) return defaultValue;
        return body.substring(q1 + 1, q2);
    }
}

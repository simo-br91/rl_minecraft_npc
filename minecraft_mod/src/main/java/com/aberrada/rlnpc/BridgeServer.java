package com.aberrada.rlnpc;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

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
        try {
            JsonObject obj = JsonParser.parseString(body).getAsJsonObject();
            return obj.get("action").getAsInt();
        } catch (Exception e) {
            return 4; // no_op fallback
        }
    }

    private String extractStringField(String body, String field, String defaultValue) {
        try {
            JsonObject obj = JsonParser.parseString(body).getAsJsonObject();
            return obj.has(field) ? obj.get(field).getAsString() : defaultValue;
        } catch (Exception e) {
            return defaultValue;
        }
    }
}

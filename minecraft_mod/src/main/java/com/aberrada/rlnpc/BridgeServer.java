package com.aberrada.rlnpc;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

/**
 * Minimal HTTP bridge exposing /health, /reset, /step, /masks.
 *
 * Endpoints:
 *   GET  /health — liveness probe
 *   POST /reset  — start a new episode
 *   POST /step   — advance one game tick
 *   GET  /masks  — returns the current 13-element action-validity mask as a
 *                  JSON array of integers (1=valid, 0=invalid). Used by
 *                  sb3-contrib MaskablePPO on the Python side. (Issue 6.3)
 *
 * JSON parsing is done once per request (was previously done 5 times).
 */
public class BridgeServer {

    private static final int PORT = 8765;

    private final EnvironmentManager env;
    private HttpServer server;

    public BridgeServer(EnvironmentManager env) {
        this.env = env;
    }

    public void start() {
        if (server != null) return;
        try {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", PORT), 0);

            server.createContext("/health", ex ->
                    sendJson(ex, 200, "{\"status\":\"ok\"}"));

            server.createContext("/reset", ex -> {
                if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                    sendJson(ex, 405, "{\"error\":\"Use POST\"}"); return;
                }
                JsonObject body = parseBody(ex);

                String  task            = getString(body, "task",              "navigation");
                boolean sparseReward    = getBool(body,   "sparse_reward",     false);
                double  minDist         = getDouble(body, "min_dist",          -1.0);
                double  maxDist         = getDouble(body, "max_dist",          -1.0);
                int     numObstacles    = getInt(body,    "num_obstacles",      -1);
                int     numCrops        = getInt(body,    "num_crops",          5);
                boolean fullFarmCycle   = getBool(body,   "full_farm_cycle",   false);
                // Combat curriculum parameters (Fix 4.5): -1 means "use defaults"
                int     numMobs         = getInt(body,    "num_mobs",          -1);
                double  mobDistMin      = getDouble(body, "mob_dist_min",      -1.0);
                double  mobDistMax      = getDouble(body, "mob_dist_max",      -1.0);

                int seed = getInt(body,"seed", -1);
                String result = env.reset(task, sparseReward, minDist, maxDist,
                                          numObstacles, numCrops, fullFarmCycle, seed,
                                          numMobs, mobDistMin, mobDistMax);
                sendJson(ex, 200, result);
            });

            server.createContext("/step", ex -> {
                if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                    sendJson(ex, 405, "{\"error\":\"Use POST\"}"); return;
                }
                JsonObject body = parseBody(ex);
                int action = getInt(body, "action", 4);
                sendJson(ex, 200, env.step(action));
            });

            // /notify — broadcasts a plain-text message to all online players.
            // Used by the Python training callbacks to show training progress
            // in-game (start, milestones, completion) without flooding chat.
            server.createContext("/notify", ex -> {
                if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                    sendJson(ex, 405, "{\"error\":\"Use POST\"}"); return;
                }
                JsonObject body = parseBody(ex);
                String message = getString(body, "message", "");
                if (!message.isEmpty()) {
                    env.broadcastMessage(message);
                }
                sendJson(ex, 200, "{\"ok\":true}");
            });

            // /masks — returns the current 13-element action validity mask.
            // Only GET is accepted (idempotent read; no body needed).
            server.createContext("/masks", ex -> {
                if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
                    sendJson(ex, 405, "{\"error\":\"Use GET\"}"); return;
                }
                sendJson(ex, 200, env.actionMasks());
            });

            server.setExecutor(null);
            server.start();
        } catch (IOException e) {
            RLNpcMod.LOGGER.error("Failed to start bridge server", e);
        }
    }

    public void stop() {
        if (server != null) { server.stop(0); server = null; }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private void sendJson(HttpExchange ex, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    private JsonObject parseBody(HttpExchange ex) {
        try (InputStream is = ex.getRequestBody()) {
            String text = new String(is.readAllBytes(), StandardCharsets.UTF_8).trim();
            if (text.isEmpty()) return new JsonObject();
            return JsonParser.parseString(text).getAsJsonObject();
        } catch (Exception e) {
            return new JsonObject();
        }
    }

    private String getString(JsonObject obj, String key, String def) {
        return (obj != null && obj.has(key)) ? obj.get(key).getAsString() : def;
    }

    private boolean getBool(JsonObject obj, String key, boolean def) {
        return (obj != null && obj.has(key)) ? obj.get(key).getAsBoolean() : def;
    }

    private int getInt(JsonObject obj, String key, int def) {
        return (obj != null && obj.has(key)) ? obj.get(key).getAsInt() : def;
    }

    private double getDouble(JsonObject obj, String key, double def) {
        return (obj != null && obj.has(key)) ? obj.get(key).getAsDouble() : def;
    }
}
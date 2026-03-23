# Bug Log

## Fixed

### BUG-001 — Agent walks through gold/emerald blocks (ghost collision)
**Symptom:** The NPC walked straight through the gold target marker and the
emerald farming marker as if they were air.  
**Root cause:** `isBlockedAt` in `ActionExecutor` checked only the entity centre
point.  With a 0.6-block-wide bounding box, the centre sits in the next block
while the corners already overlap the marker block — the single-point check
returned false.  
**Fix:** Multi-point footprint — 5 sample points (centre + 4 corners at ±0.27 in
X and Z). Any corner entering a solid block triggers a collision.  
**File:** `ActionExecutor.java` — `isBlockedAt()`, `checkPoint()`

---

### BUG-002 — Jump action never used during training
**Symptom:** In thousands of training episodes the agent never selected action 5
(jump). TensorBoard showed action 5 probability converging to ~0.  
**Root cause A:** The flat test world had zero obstacles, so the jump action had
zero expected value — PPO correctly learned to ignore it.  
**Root cause B:** The original jump implementation used `setDeltaMovement`, which
the game tick overrode on the very next tick because `moveTo()` resets
velocity — the agent never actually moved upward.  
**Fix A:** `EnvironmentManager.placeNavigationObstacles()` now places 1–2 single
stone blocks directly on the spawn-to-target path at each reset, so jumping is
always required to complete navigation episodes.  
**Fix B:** Jump redesigned as a deterministic teleport-up-1-block action with
preconditions: blocked at current height, clear one block higher, headroom above
agent. The agent then walks forward normally and `findSurfaceY` snaps it back to
the ground when it steps off the wall.  
**Files:** `ActionExecutor.java` — `jumpUp()`, `EnvironmentManager.java` —
`placeNavigationObstacles()`

---

### BUG-003 — Multiple gold blocks left in world across resets
**Symptom:** After a few resets the world accumulated gold blocks at old target
positions.  
**Root cause:** `clearTaskArtifacts()` was not called reliably before
`placeMarker()`, and the marker position from the previous episode was sometimes
`null`.  
**Fix:** `clearTaskArtifacts()` now always clears `state.markerPos` if non-null,
and all obstacle positions are stored in `state.obstaclePositions` (cleared every
reset).  
**File:** `EnvironmentManager.java`

---

### BUG-004 — Jump behaved like upward teleportation / "flying"
**Symptom:** Repeating the jump action made the agent float upward indefinitely
because there was no gravity and no surface-snap after jump.  
**Root cause:** No preconditions on jump + no surface-snap after landing.  
**Fix:** Jump now requires (1) something blocking directly ahead, (2) clear
space one block above the obstacle, (3) headroom above the agent. After jumping
the agent is at +1Y but still on top of the wall; `findSurfaceY` in the next
`moveForward` call snaps them back to the ground surface on the far side.  
**File:** `ActionExecutor.java` — `jumpUp()`, `findSurfaceY()`

---

### BUG-005 — Thread-safety race between HTTP handler and game tick
**Symptom:** Intermittent `ConcurrentModificationException` in server logs;
occasional stale observations (obs reflected the state from the previous step).  
**Root cause:** `EnvironmentManager.reset()` and `.step()` were calling
`level.setBlockAndUpdate()` and `agent.moveTo()` directly from the HTTP server
thread — these methods are not thread-safe and must run on the main server tick
thread.  
**Fix:** All world/entity manipulation wrapped in `server.execute(() -> { ... })`
with `CompletableFuture` + `future.get(timeout)` on the HTTP thread.  
**File:** `EnvironmentManager.java`

---

## Known limitations / open issues

- **Jump produces a one-step "float"** at the top of the obstacle before
  `findSurfaceY` snaps back to ground. Cosmetically odd but functionally correct.
- **Interact range** is a single block directly ahead at 0.9-block distance.
  Approaching from an angle can miss the crop even if the agent appears close.
- **Farming only supports harvest**, not full plant-then-harvest. The crop is
  always pre-grown at episode start. Full farming cycle is a future extension.
- **No multi-environment parallelism** — SB3 `SubprocVecEnv` is not possible
  with a single Minecraft instance. Training is single-env, which limits
  throughput.

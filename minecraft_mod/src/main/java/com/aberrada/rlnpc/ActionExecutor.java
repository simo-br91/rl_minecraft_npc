package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.level.block.state.BlockState;

import java.util.List;

/**
 * Applies discrete actions to the RL NPC entity.
 *
 * Action Space (13 actions):
 *   0  forward (walk)
 *   1  turn_left
 *   2  turn_right
 *   3  interact  (harvest crop / use bonemeal / till soil)
 *   4  no_op
 *   5  jump
 *   6  sprint_forward (faster forward, costs more food)
 *   7  move_backward
 *   8  strafe_left
 *   9  strafe_right
 *  10  attack        (melee swing at nearest mob)
 *  11  eat           (eat food from inventory)
 *  12  switch_item   (cycle active inventory slot)
 *
 * Pitch (vertical look) is controlled autonomously each step to make
 * movement look human-like — it smoothly looks toward targets.
 */
public class ActionExecutor {

    // Movement constants
    private static final float   TURN_DEGREES   = 15.0f;
    private static final double  WALK_STEP      = 0.35;
    private static final double  SPRINT_STEP    = 0.55;
    private static final double  BACKWARD_STEP  = 0.20;
    private static final double  STRAFE_STEP    = 0.25;
    private static final double  HALF_WIDTH     = 0.27;

    // Pitch smoothing
    private static final float   PITCH_SMOOTH   = 0.15f;   // blend factor per step
    private static final float   PITCH_MAX      = 45.0f;
    private static final float   PITCH_MIN      = -30.0f;

    // Combat
    private static final double  ATTACK_RANGE   = 3.5;

    // ------------------------------------------------------------------
    // Entry point
    // ------------------------------------------------------------------

    public static boolean applyAction(RLNpcEntity agent, int action, EpisodeState state) {
        boolean valid = switch (action) {
            case 0  -> moveForward(agent, WALK_STEP);
            case 1  -> turnLeft(agent);
            case 2  -> turnRight(agent);
            case 3  -> true;   // interact handled externally in EnvironmentManager
            case 4  -> true;   // no-op
            case 5  -> jumpUp(agent, state);
            case 6  -> sprintForward(agent, state);
            case 7  -> moveBackward(agent);
            case 8  -> strafeLeft(agent);
            case 9  -> strafeRight(agent);
            case 10 -> attack(agent, state);
            case 11 -> true;   // eat handled externally
            case 12 -> switchItem(agent, state);
            default -> false;
        };
        // Smooth pitch toward a natural angle after each action
        updatePitch(agent, state);
        return valid;
    }

    // ------------------------------------------------------------------
    // Movement
    // ------------------------------------------------------------------

    private static boolean moveForward(RLNpcEntity agent, double step) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;
        double tx = agent.getX() + flat.x * step;
        double tz = agent.getZ() + flat.z * step;
        return applyHorizontalMove(agent, tx, tz);
    }

    private static boolean sprintForward(RLNpcEntity agent, EpisodeState state) {
        state.lastSprinting = true;
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;
        double tx = agent.getX() + flat.x * SPRINT_STEP;
        double tz = agent.getZ() + flat.z * SPRINT_STEP;
        return applyHorizontalMove(agent, tx, tz);
    }

    private static boolean moveBackward(RLNpcEntity agent) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;
        double tx = agent.getX() - flat.x * BACKWARD_STEP;
        double tz = agent.getZ() - flat.z * BACKWARD_STEP;
        return applyHorizontalMove(agent, tx, tz);
    }

    private static boolean strafeLeft(RLNpcEntity agent) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;
        // Perpendicular left: rotate flat 90° CCW → (-z, x)
        double tx = agent.getX() - flat.z * STRAFE_STEP;
        double tz = agent.getZ() + flat.x * STRAFE_STEP;
        return applyHorizontalMove(agent, tx, tz);
    }

    private static boolean strafeRight(RLNpcEntity agent) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;
        // Perpendicular right: rotate flat 90° CW → (z, -x)
        double tx = agent.getX() + flat.z * STRAFE_STEP;
        double tz = agent.getZ() - flat.x * STRAFE_STEP;
        return applyHorizontalMove(agent, tx, tz);
    }

    private static boolean applyHorizontalMove(RLNpcEntity agent, double tx, double tz) {
        double cy = agent.getY();
        if (isBlockedAt(agent, tx, cy, tz)) return false;
        double landY = findSurfaceY(agent, tx, cy, tz);
        float  yaw   = agent.getYRot();
        agent.moveTo(tx, landY, tz, yaw, agent.getXRot());
        syncRotations(agent, yaw, agent.getXRot());
        return true;
    }

    private static boolean turnLeft(RLNpcEntity agent) {
        return rotateYaw(agent, agent.getYRot() - TURN_DEGREES);
    }

    private static boolean turnRight(RLNpcEntity agent) {
        return rotateYaw(agent, agent.getYRot() + TURN_DEGREES);
    }

    private static boolean rotateYaw(RLNpcEntity agent, float newYaw) {
        agent.moveTo(agent.getX(), agent.getY(), agent.getZ(), newYaw, agent.getXRot());
        syncRotations(agent, newYaw, agent.getXRot());
        return true;
    }

    /**
     * Jump: teleport up 1 block when a 1-block wall is ahead.
     * After landing, findSurfaceY in the next move snaps back down.
     */
    private static boolean jumpUp(RLNpcEntity agent, EpisodeState state) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;
        double x = agent.getX(), y = agent.getY(), z = agent.getZ();
        double tx = x + flat.x * WALK_STEP;
        double tz = z + flat.z * WALK_STEP;
        if (!isBlockedAt(agent, tx, y, tz)) return false;
        if (isBlockedAt(agent, tx, y + 1.0, tz)) return false;
        if (isBlockedAt(agent, x, y + 1.0, z)) return false;
        agent.moveTo(x, y + 1.0, z, agent.getYRot(), agent.getXRot());
        syncRotations(agent, agent.getYRot(), agent.getXRot());
        if (state != null) state.lastJumpedObstacle = true;
        return true;
    }

    // ------------------------------------------------------------------
    // Combat
    // ------------------------------------------------------------------

    /**
     * Swing at the nearest hostile mob within ATTACK_RANGE.
     * Uses Minecraft's standard hurt() mechanism.
     */
    public static boolean attack(RLNpcEntity agent, EpisodeState state) {
        AABB searchBox = agent.getBoundingBox().inflate(ATTACK_RANGE);
        List<Monster> mobs = agent.level().getEntitiesOfClass(
                Monster.class, searchBox,
                m -> m.isAlive() && !m.isSpectator());
        if (mobs.isEmpty()) return false;

        // Find closest
        Monster target = null;
        double bestDist = Double.MAX_VALUE;
        for (Monster m : mobs) {
            double d = m.distanceTo(agent);
            if (d < bestDist) { bestDist = d; target = m; }
        }
        if (target == null) return false;

        // Face the target for realism
        faceEntity(agent, target);

        // Apply damage — 6.0 = iron sword base damage
        float damage = 6.0f;
        boolean killed = false;
        target.hurt(agent.damageSources().mobAttack(agent), damage);
        if (!target.isAlive()) {
            killed = true;
            if (state != null) state.mobsKilled++;
        }
        if (state != null) state.lastAttackValid = true;
        return true;
    }

    // ------------------------------------------------------------------
    // Inventory
    // ------------------------------------------------------------------

    private static boolean switchItem(RLNpcEntity agent, EpisodeState state) {
        if (state == null) return false;
        state.activeSlot = (state.activeSlot + 1) % 5;  // 5 tracked slots
        return true;
    }

    // ------------------------------------------------------------------
    // Pitch (vertical look) — smoothed for human-like appearance
    // ------------------------------------------------------------------

    /**
     * Smoothly blend the current pitch toward targetPitch.
     * targetPitch is set by EnvironmentManager based on situation
     * (looking at crop, looking at mob, looking forward).
     */
    public static void updatePitch(RLNpcEntity agent, EpisodeState state) {
        if (state == null) return;
        float current = state.currentPitch;
        float target  = Math.max(PITCH_MIN, Math.min(PITCH_MAX, state.targetPitch));
        float newPitch = current + (target - current) * PITCH_SMOOTH;
        state.currentPitch = newPitch;
        syncRotations(agent, agent.getYRot(), newPitch);
    }

    // ------------------------------------------------------------------
    // Collision helpers
    // ------------------------------------------------------------------

    public static boolean isBlockedAt(RLNpcEntity agent, double x, double y, double z) {
        double r = HALF_WIDTH;
        return checkPoint(agent, x,   y, z)
            || checkPoint(agent, x+r, y, z+r)
            || checkPoint(agent, x+r, y, z-r)
            || checkPoint(agent, x-r, y, z+r)
            || checkPoint(agent, x-r, y, z-r);
    }

    private static boolean checkPoint(RLNpcEntity agent, double x, double y, double z) {
        BlockPos feet = BlockPos.containing(x, y, z);
        BlockPos head = feet.above();
        BlockState feetState = agent.level().getBlockState(feet);
        BlockState headState = agent.level().getBlockState(head);
        return solidBlock(feetState) || solidBlock(headState);
    }

    private static boolean solidBlock(BlockState s) {
        return !s.isAir() && s.blocksMotion();
    }

    // ------------------------------------------------------------------
    // Surface snap — scan up to 5 blocks down
    // ------------------------------------------------------------------

    public static double findSurfaceY(RLNpcEntity agent, double x, double startY, double z) {
        var level = agent.level();
        int ix = (int) Math.floor(x);
        int iz = (int) Math.floor(z);
        int iy = (int) Math.floor(startY);
        for (int dy = 0; dy >= -5; dy--) {
            BlockPos ground = new BlockPos(ix, iy + dy, iz);
            BlockState gs   = level.getBlockState(ground);
            if (!gs.isAir() && gs.blocksMotion()) {
                return ground.getY() + 1.0;
            }
        }
        return startY;
    }

    // ------------------------------------------------------------------
    // Geometry helpers
    // ------------------------------------------------------------------

    public static Vec3 getHorizontalLook(RLNpcEntity agent) {
        Vec3 look = agent.getLookAngle();
        Vec3 flat = new Vec3(look.x, 0.0, look.z);
        if (flat.lengthSqr() < 1e-8) return null;
        return flat.normalize();
    }

    public static BlockPos frontBlockPos(RLNpcEntity agent) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) flat = new Vec3(0, 0, 1);
        Vec3 origin = agent.position().add(0.0, 0.2, 0.0);
        Vec3 front  = origin.add(flat.scale(0.9));
        return BlockPos.containing(front.x, origin.y, front.z);
    }

    /** Turn the agent to face a target entity. */
    private static void faceEntity(RLNpcEntity agent, LivingEntity target) {
        double dx = target.getX() - agent.getX();
        double dy = target.getEyeY() - agent.getEyeY();
        double dz = target.getZ() - agent.getZ();
        double hDist = Math.sqrt(dx * dx + dz * dz);
        float  yaw   = (float)(Math.toDegrees(Math.atan2(dz, dx))) - 90.0f;
        float  pitch = (float)(-Math.toDegrees(Math.atan2(dy, hDist)));
        pitch = Math.max(PITCH_MIN, Math.min(PITCH_MAX, pitch));
        syncRotations(agent, yaw, pitch);
    }

    static void syncRotations(RLNpcEntity agent, float yaw, float pitch) {
        agent.setYRot(yaw);
        agent.setXRot(pitch);
        agent.setYHeadRot(yaw);
        agent.yBodyRot = yaw;
        agent.yHeadRot = yaw;
    }
}
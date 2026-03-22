package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

public class ActionExecutor {

    private static final float   TURN_DEGREES      = 15.0f;
    private static final double  FORWARD_STEP      = 0.35;
    // Half the entity's collision width (entity is 0.6 wide → 0.3 each side).
    // Use 0.27 so corner probes stay safely inside block boundaries.
    private static final double  HALF_WIDTH        = 0.27;

    // -----------------------------------------------------------------------
    // Public entry point
    // -----------------------------------------------------------------------

    /**
     * Apply one discrete action to the agent.
     * state may be null (used only for jump bookkeeping).
     */
    public static boolean applyMovementAction(RLNpcEntity agent, int action, EpisodeState state) {
        return switch (action) {
            case 0 -> moveForward(agent);
            case 1 -> turnLeft(agent);
            case 2 -> turnRight(agent);
            case 4 -> true;                         // no-op
            case 5 -> jumpUp(agent, state);         // jump-up-in-place to clear 1-block walls
            default -> false;
        };
    }

    // -----------------------------------------------------------------------
    // Movement actions
    // -----------------------------------------------------------------------

    private static boolean moveForward(RLNpcEntity agent) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;

        double targetX = agent.getX() + flat.x * FORWARD_STEP;
        double targetZ = agent.getZ() + flat.z * FORWARD_STEP;
        double currentY = agent.getY();

        if (isBlockedAt(agent, targetX, currentY, targetZ)) {
            return false;   // solid wall in front — agent must jump first
        }

        // Snap to surface so the agent follows 1-block-high steps naturally
        double landY = findSurfaceY(agent, targetX, currentY, targetZ);
        float  yaw   = agent.getYRot();
        agent.moveTo(targetX, landY, targetZ, yaw, 0.0f);
        syncRotations(agent, yaw);
        return true;
    }

    private static boolean turnLeft(RLNpcEntity agent) {
        return rotateInPlace(agent, agent.getYRot() - TURN_DEGREES);
    }

    private static boolean turnRight(RLNpcEntity agent) {
        return rotateInPlace(agent, agent.getYRot() + TURN_DEGREES);
    }

    private static boolean rotateInPlace(RLNpcEntity agent, float newYaw) {
        agent.moveTo(agent.getX(), agent.getY(), agent.getZ(), newYaw, 0.0f);
        syncRotations(agent, newYaw);
        return true;
    }

    /**
     * Jump action: teleport the agent straight UP by 1 block when there is a
     * 1-block-high obstacle directly ahead.  The forward motion then happens
     * on the NEXT forward action, which will glide the agent across the wall
     * top and snap back to ground level on the far side via findSurfaceY.
     *
     * This is the correct design for a teleport-based agent: real physics
     * jumping is unreliable because our moveTo() calls override velocity every
     * step.
     */
    private static boolean jumpUp(RLNpcEntity agent, EpisodeState state) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) return false;

        double x = agent.getX();
        double y = agent.getY();
        double z = agent.getZ();

        double targetX = x + flat.x * FORWARD_STEP;
        double targetZ = z + flat.z * FORWARD_STEP;

        // Require: something is blocking us at current height
        if (!isBlockedAt(agent, targetX, y, targetZ)) return false;

        // Require: one block up is clear (1-block wall, not a 2-block wall)
        if (isBlockedAt(agent, targetX, y + 1.0, targetZ)) return false;

        // Require: headroom directly above the agent
        if (isBlockedAt(agent, x, y + 1.0, z)) return false;

        // Teleport up 1 block in place (agent then walks forward on next step)
        agent.moveTo(x, y + 1.0, z, agent.getYRot(), 0.0f);
        syncRotations(agent, agent.getYRot());

        if (state != null) state.lastJumpedObstacle = true;
        return true;
    }

    // -----------------------------------------------------------------------
    // Collision helpers
    // -----------------------------------------------------------------------

    /**
     * Checks whether placing the entity centre at (x, y, z) would cause the
     * entity's bounding box to intersect a solid block.
     *
     * We sample the four corners of the entity footprint plus the centre.
     * This catches cases where the centre is still in clear air but a bounding-
     * box corner has already entered a block — the main cause of ghost-through-
     * block behaviour with the old single-point check.
     */
    public static boolean isBlockedAt(RLNpcEntity agent, double x, double y, double z) {
        double r = HALF_WIDTH;
        // 5-point footprint: centre + 4 corners
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
        return (solidBlock(feetState) || solidBlock(headState));
    }

    private static boolean solidBlock(BlockState s) {
        return !s.isAir() && s.blocksMotion();
    }

    // -----------------------------------------------------------------------
    // Surface snap
    // -----------------------------------------------------------------------

    /**
     * Given a candidate (x, startY, z), scan downward up to 3 blocks to find
     * the first solid surface and return the Y the agent should stand at.
     * This lets moveForward naturally step DOWN off wall tops or ramps without
     * extra actions.
     */
    private static double findSurfaceY(RLNpcEntity agent, double x, double startY, double z) {
        var level = agent.level();
        int ix = (int) Math.floor(x);
        int iz = (int) Math.floor(z);
        int iy = (int) Math.floor(startY);

        for (int dy = 0; dy >= -3; dy--) {
            BlockPos ground = new BlockPos(ix, iy + dy, iz);
            BlockState gs    = level.getBlockState(ground);
            if (!gs.isAir() && gs.blocksMotion()) {
                // Stand on top of this block
                return ground.getY() + 1.0;
            }
        }
        return startY;  // fallback: no surface found, keep current Y
    }

    // -----------------------------------------------------------------------
    // Geometry helpers
    // -----------------------------------------------------------------------

    /** Returns the normalised horizontal look vector, or null if degenerate. */
    static Vec3 getHorizontalLook(RLNpcEntity agent) {
        Vec3 look = agent.getLookAngle();
        Vec3 flat = new Vec3(look.x, 0.0, look.z);
        if (flat.lengthSqr() < 1e-8) return null;
        return flat.normalize();
    }

    /** Front block position used by ObservationBuilder (kept here to avoid duplication). */
    public static BlockPos frontBlockPos(RLNpcEntity agent) {
        Vec3 flat = getHorizontalLook(agent);
        if (flat == null) flat = new Vec3(0, 0, 1);
        Vec3 origin = agent.position().add(0.0, 0.2, 0.0);
        Vec3 front  = origin.add(flat.scale(0.9));
        return BlockPos.containing(front.x, origin.y, front.z);
    }

    private static void syncRotations(RLNpcEntity agent, float yaw) {
        agent.setYRot(yaw);
        agent.setYHeadRot(yaw);
        agent.yBodyRot = yaw;
        agent.yHeadRot = yaw;
    }
}

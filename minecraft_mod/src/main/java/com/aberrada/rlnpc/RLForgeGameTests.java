package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTest;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraftforge.gametest.GameTestHolder;
import net.minecraftforge.gametest.PrefixGameTestTemplate;

/**
 * Forge game tests for the RL NPC mod.
 *
 * These tests run server-side in the game test environment (runGameTestServer).
 * They verify that the core mod components work correctly without requiring
 * a Python client or HTTP requests.
 *
 * Run via: ./gradlew runGameTestServer
 *
 * Implements Issue 3.10 — Forge game tests were listed as a missing item.
 */
@GameTestHolder(RLNpcMod.MOD_ID)
@PrefixGameTestTemplate(false)
public class RLForgeGameTests {

    // Default structure name — uses an empty 3×3×3 air structure
    private static final String EMPTY_STRUCTURE = "rlnpc:empty";

    // ------------------------------------------------------------------
    // Entity registration
    // ------------------------------------------------------------------

    /**
     * Verify that the RLNpcEntity type is registered and can be instantiated.
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testRLNpcEntityCreation(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "RLNpcEntity should be created successfully");
        helper.assertTrue(entity instanceof RLNpcEntity,
                "Created entity should be an instance of RLNpcEntity");
        if (entity != null) {
            // Clean up
            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }

    /**
     * Verify that the NPC entity has the correct attributes.
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testRLNpcAttributes(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "Entity must be non-null");
        if (entity != null) {
            var maxHealth = entity.getAttribute(
                    net.minecraft.world.entity.ai.attributes.Attributes.MAX_HEALTH);
            helper.assertTrue(maxHealth != null, "MAX_HEALTH attribute must exist");
            helper.assertTrue(maxHealth.getBaseValue() == 20.0,
                    "MAX_HEALTH must be 20.0, got " + maxHealth.getBaseValue());
            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }

    /**
     * Verify that AI is disabled on the NPC (it should not path-find).
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testRLNpcNoAI(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "Entity must be non-null");
        if (entity != null) {
            helper.assertTrue(entity.isNoAi(), "RLNpcEntity must have AI disabled");
            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }

    // ------------------------------------------------------------------
    // Observation builder
    // ------------------------------------------------------------------

    /**
     * Verify that ObservationBuilder.build() returns a vector of the expected dimension.
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testObservationDimension(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "Entity must be non-null");
        if (entity != null) {
            BlockPos spawnPos = helper.absolutePos(new BlockPos(0, 1, 0));
            level.addFreshEntity(entity);
            entity.moveTo(spawnPos.getX() + 0.5, spawnPos.getY(), spawnPos.getZ() + 0.5,
                          0.0f, 0.0f);

            EpisodeState state = new EpisodeState();
            state.targetX = spawnPos.getX() + 5.0;
            state.targetZ = spawnPos.getZ();

            double[] obs = ObservationBuilder.build(entity, state);
            helper.assertTrue(obs.length == ObservationBuilder.OBS_DIM,
                    "Observation dimension should be " + ObservationBuilder.OBS_DIM
                    + " but got " + obs.length);

            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }

    /**
     * Verify that all observation values are within expected bounds.
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testObservationBounds(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "Entity must be non-null");
        if (entity != null) {
            BlockPos spawnPos = helper.absolutePos(new BlockPos(0, 1, 0));
            level.addFreshEntity(entity);
            entity.moveTo(spawnPos.getX() + 0.5, spawnPos.getY(), spawnPos.getZ() + 0.5,
                          0.0f, 0.0f);

            EpisodeState state = new EpisodeState();
            state.targetX = spawnPos.getX() + 8.0;
            state.targetZ = spawnPos.getZ();

            double[] obs = ObservationBuilder.build(entity, state);
            for (int i = 0; i < obs.length; i++) {
                helper.assertTrue(obs[i] >= -1.0 && obs[i] <= 1.0,
                        "Observation index " + i + " = " + obs[i]
                        + " is out of bounds [-1, 1]");
            }
            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }

    // ------------------------------------------------------------------
    // Action executor
    // ------------------------------------------------------------------

    /**
     * Verify that turn_left and turn_right change the entity's yaw.
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testTurnActions(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "Entity must be non-null");
        if (entity != null) {
            BlockPos spawnPos = helper.absolutePos(new BlockPos(0, 1, 0));
            level.addFreshEntity(entity);
            entity.moveTo(spawnPos.getX() + 0.5, spawnPos.getY(), spawnPos.getZ() + 0.5,
                          0.0f, 0.0f);

            float initialYaw = entity.getYRot();
            EpisodeState state = new EpisodeState();

            // Turn left (-15°)
            ActionExecutor.applyAction(entity, 1, state);
            helper.assertTrue(Math.abs(entity.getYRot() - (initialYaw - 15.0f)) < 0.01f,
                    "Turn left should decrease yaw by 15°");

            // Turn right (+15° × 2 from initial to net +15°)
            ActionExecutor.applyAction(entity, 2, state);
            ActionExecutor.applyAction(entity, 2, state);
            helper.assertTrue(Math.abs(entity.getYRot() - (initialYaw + 15.0f)) < 0.01f,
                    "Net yaw should be +15° after left+right+right");

            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }

    // ------------------------------------------------------------------
    // Reward calculator
    // ------------------------------------------------------------------

    /**
     * Verify that sparse reward returns SPARSE_SUCCESS (+10) on success.
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testSparseRewardSuccess(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "Entity must be non-null");
        if (entity != null) {
            level.addFreshEntity(entity);
            EpisodeState state = new EpisodeState();
            state.sparseReward = true;

            double reward = RewardCalculator.compute(
                    entity, state, 0, 5.0, 4.5, true, true, false);
            helper.assertTrue(reward == 10.0,
                    "Sparse success reward should be 10.0 but got " + reward);

            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }

    /**
     * Verify that shaped reward gives a positive value when closing distance.
     */
    @GameTest(template = EMPTY_STRUCTURE)
    public static void testShapedRewardProgress(GameTestHelper helper) {
        var level = helper.getLevel();
        var entity = ModEntities.RL_NPC.get().create(level);
        helper.assertTrue(entity != null, "Entity must be non-null");
        if (entity != null) {
            level.addFreshEntity(entity);
            EpisodeState state = new EpisodeState();
            state.sparseReward = false;
            // before=10, after=9 → progress +1 → 0.15*1 - 0.015 = +0.135
            double reward = RewardCalculator.compute(
                    entity, state, 0, 10.0, 9.0, true, false, false);
            helper.assertTrue(reward > 0,
                    "Shaped reward should be positive when closing distance, got " + reward);

            entity.remove(net.minecraft.world.entity.Entity.RemovalReason.DISCARDED);
        }
        helper.succeed();
    }
}

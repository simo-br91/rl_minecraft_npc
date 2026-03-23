package com.aberrada.rlnpc;

import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.SwordItem;

/**
 * Manages the RL NPC's inventory and equipment.
 *
 * Fixed inventory layout:
 *   Slot 0 (SWORD_SLOT)  — Iron Sword (main weapon)
 *   Slot 1 (FOOD_SLOT)   — Cooked Beef × 8 (food)
 *   Slot 2 (SEEDS_SLOT)  — Wheat Seeds × 16 (farming)
 *   Slot 3 (HOE_SLOT)    — Iron Hoe (tilling)
 *   Slot 4 (BONES_SLOT)  — Bone Meal × 8 (growing crops)
 *
 * The agent can switch between slots (action 12) and use the
 * held item for context-specific actions.
 */
public class InventoryManager {

    public static final int SLOT_SWORD = EpisodeState.SLOT_SWORD;
    public static final int SLOT_FOOD  = EpisodeState.SLOT_FOOD;
    public static final int SLOT_SEEDS = EpisodeState.SLOT_SEEDS;
    public static final int SLOT_HOE   = EpisodeState.SLOT_HOE;
    public static final int SLOT_BONES = EpisodeState.SLOT_BONES;

    private static final int FOOD_STACK_SIZE   = 8;
    private static final int SEEDS_STACK_SIZE  = 16;
    private static final int BONEMEAL_STACK    = 8;

    /**
     * Give the agent a full starting inventory and equip armor.
     * Called once at the start of every episode.
     */
    public static void equipAgent(RLNpcEntity agent, EpisodeState state) {
        // Clear existing inventory
        agent.getInventory().clearContent();

        // Slot 0: Iron Sword (main hand for combat)
        agent.getInventory().setItem(SLOT_SWORD, new ItemStack(Items.IRON_SWORD));
        // Slot 1: Cooked Beef (food)
        ItemStack food = new ItemStack(Items.COOKED_BEEF, FOOD_STACK_SIZE);
        agent.getInventory().setItem(SLOT_FOOD, food);
        // Slot 2: Wheat Seeds
        agent.getInventory().setItem(SLOT_SEEDS, new ItemStack(Items.WHEAT_SEEDS, SEEDS_STACK_SIZE));
        // Slot 3: Iron Hoe (for tilling in full farming cycle)
        agent.getInventory().setItem(SLOT_HOE, new ItemStack(Items.IRON_HOE));
        // Slot 4: Bone Meal
        agent.getInventory().setItem(SLOT_BONES, new ItemStack(Items.BONE_MEAL, BONEMEAL_STACK));

        // Iron armor for survivability
        agent.setItemSlot(EquipmentSlot.HEAD,  new ItemStack(Items.IRON_HELMET));
        agent.setItemSlot(EquipmentSlot.CHEST, new ItemStack(Items.IRON_CHESTPLATE));
        agent.setItemSlot(EquipmentSlot.LEGS,  new ItemStack(Items.IRON_LEGGINGS));
        agent.setItemSlot(EquipmentSlot.FEET,  new ItemStack(Items.IRON_BOOTS));

        // Set active slot to sword initially
        state.activeSlot = SLOT_SWORD;
        syncMainHandFromSlot(agent, state);
    }

    /**
     * Sync the entity's main-hand item with the active state slot.
     * Minecraft PathfinderMob uses getMainHandItem() for various checks.
     */
    public static void syncMainHandFromSlot(RLNpcEntity agent, EpisodeState state) {
        ItemStack held = agent.getInventory().getItem(state.activeSlot);
        agent.setItemSlot(EquipmentSlot.MAINHAND, held.copy());
    }

    /**
     * Eat food from slot 1. Returns true if food was available and consumed.
     * Restores 8 hunger and 12.8 saturation (cooked beef values).
     */
    public static boolean eatFood(RLNpcEntity agent, EpisodeState state) {
        if (state.foodLevel >= 20) return false;   // not hungry
        ItemStack food = agent.getInventory().getItem(SLOT_FOOD);
        if (food.isEmpty()) return false;

        // Simulate eating: restore food
        state.foodLevel = Math.min(20, state.foodLevel + 8);
        state.saturation = Math.min(20.0f, state.saturation + 12.8f);
        food.shrink(1);
        agent.getInventory().setItem(SLOT_FOOD, food);
        state.lastEatValid = true;
        return true;
    }

    /**
     * Returns true if the agent is currently holding a sword.
     */
    public static boolean isHoldingSword(RLNpcEntity agent, EpisodeState state) {
        return state.activeSlot == SLOT_SWORD;
    }

    /**
     * Returns the item name in the active slot (for info dict).
     */
    public static String getActiveItemName(EpisodeState state) {
        return switch (state.activeSlot) {
            case SLOT_SWORD -> "iron_sword";
            case SLOT_FOOD  -> "cooked_beef";
            case SLOT_SEEDS -> "wheat_seeds";
            case SLOT_HOE   -> "iron_hoe";
            case SLOT_BONES -> "bone_meal";
            default         -> "empty";
        };
    }

    /**
     * Returns true if agent has bonemeal (slot 4 not empty).
     */
    public static boolean hasBonemeal(RLNpcEntity agent) {
        return !agent.getInventory().getItem(SLOT_BONES).isEmpty();
    }

    /**
     * Consume one bonemeal from slot 4.
     */
    public static boolean consumeBonemeal(RLNpcEntity agent) {
        ItemStack bones = agent.getInventory().getItem(SLOT_BONES);
        if (bones.isEmpty()) return false;
        bones.shrink(1);
        agent.getInventory().setItem(SLOT_BONES, bones);
        return true;
    }
}
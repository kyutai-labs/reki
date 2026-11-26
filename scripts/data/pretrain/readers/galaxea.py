"""Reader for the Galaxea Open-World Dataset (R1 Lite, 227 tarballs at the pin): one leaf.

Each tarball is one LeRobot v2.1 dataset of one task session; tarballs are read in name order,
episodes in meta order. Per tarball: a state/action schema other than ``STATE_LAYOUT`` /
``ACTION_LAYOUT`` skips the whole tarball (``SCHEMA_SKIPPED``, 7-dim arms). Per episode:

- upstream quality filter: an episode whose ``quality_index`` or ``coarse_quality_index``
  points at ``unqualified`` / ``null`` is left out (operator mistakes, bad topic rates);
- state (64) / action (26): the named columns concatenated in layout order;
- texts: the English side of upstream's ``中文@English`` strings, then the text rules
  (``common.tasks``). The episode task is the coarse task; ``task_index`` is per frame, the
  fine subtask (the coarse task where the fine one is empty or a quality label);
- instruction segments: runs of equal fine subtask (before the text rules), for the lookup;
- videos: upstream mp4s, stream-copied. An episode with a video at another rate than the leaf
  (15 fps) is left out and logged; the pin lists them (``FPS_SKIPPED``), and any other one
  raises.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..common import lookup, v30
from ..common.download import untar
from ..common.stats import op
from ..common.tasks import TextRules
from ..common.video import ffprobe
from ..pins import HfSnapshot, Pin
from ..recipe import Recipe

logger = logging.getLogger(__name__)

NAME = "galaxea"
REPO = "OpenGalaxea/Galaxea-Open-World-Dataset"
REVISION = "df670d8dc3f2b369f55d28a3428b8639f0dab37d"
LEAF = "galaxea_lerobot_v30"
FPS = 15
ROBOT_TYPE = "r1lite"

TARBALLS = (
    "Adjust_The_Air_Conditioner_Temperature_20250711_006.tar.gz",
    "Arrange_Fruits_20250819_011.tar.gz",
    "Arrange_Mineral_Water_Bottles_On_The_Desk_20250721_008.tar.gz",
    "Arrange_Sofa_Cushions_20250708_006.tar.gz",
    "Arrange_Sofa_Cushions_20250804_012.tar.gz",
    "Arrange_The_Fruits_20250716_006.tar.gz",
    "Arrange_The_Fruits_20250731_012.tar.gz",
    "Arrange_The_Fruits_20250822_013.tar.gz",
    "Arrange_The_Sofa_Cushions_20250722_008.tar.gz",
    "Arrange_The_Tray_20250725_010.tar.gz",
    "Arrange_The_Tray_20250804_012.tar.gz",
    "Arrange_The_Tray_20250822_013.tar.gz",
    "Arrange_Throw_Pillows_On_Living_Room_Sofa_20250624_002.tar.gz",
    "Arrange_Toiletries_20250731_012.tar.gz",
    "Bedside_Console_With_Switch_Panel20250624_002.tar.gz",
    "Boil_The_Water_20250714_006.tar.gz",
    "Boil_The_Water_20250801_012.tar.gz",
    "Chair_Push_And_Place_20250709_006.tar.gz",
    "Chair_Push_And_Place_20250719_008.tar.gz",
    "Chair_Push_And_Place_20250801_012.tar.gz",
    "Check_Out_At_Self-checkout_Counter_20250607_003.tar.gz",
    "Clean_Cutting_Board_20250627_002.tar.gz",
    "Clean_The_Chopping_Board_20250722_008.tar.gz",
    "Clean_The_Cutting_Board_20250711_006.tar.gz",
    "Clean_The_Cutting_Board_20250804_012.tar.gz",
    "Clean_The_Mirror_20250714_006.tar.gz",
    "Clean_The_Mirror_20250731_012.tar.gz",
    "Clean_The_Sink_20250710_006.tar.gz",
    "Clean_The_Sink_20250721_008.tar.gz",
    "Clean_The_Sink_20250730_012.tar.gz",
    "Clean_The_Sofa20250627_002.tar.gz",
    "Clean_The_Toilet20250618_001.tar.gz",
    "Clean_Toilet_20250625_002.tar.gz",
    "Clothes_Into_The_Basket_20250728_010.tar.gz",
    "Collect_Juice_20250621_001.tar.gz",
    "Connect_Router_Cables_20250625_002.tar.gz",
    "Cooking_Food_In_An_Air_Fryer_20250719_008.tar.gz",
    "Cooking_Food_In_An_Air_Fryer_20250801_012.tar.gz",
    "Desktop_Garbage_Decluttering_20250712_006.tar.gz",
    "Desktop_Garbage_Organizing20250721_008.tar.gz",
    "Desktop_Trash_Disposal_20250729_011.tar.gz",
    "Dispose_Of_Garbage_In_The_Trash_Can.tar.gz",
    "Dispose_Of_Leftover_Food_20250626_002.tar.gz",
    "Doll_Storage_20250731_012.tar.gz",
    "Dry_Clothes_In_A_Dryer_20250619_001.tar.gz",
    "Egg_Placement20250703_002.tar.gz",
    "Enter_Exit_Door_Turn_On_Off_lights_20250620_001.tar.gz",
    "Floor_cloth_wiping_stains_20250614_001.tar.gz",
    "Flowers_Are_Inserted_Into_Vases_20250725_010.tar.gz",
    "Fold_Clothes20250617_001.tar.gz",
    "Fold_Clothes_20250807_011.tar.gz",
    "From_Kitchen_To_Bedroom20250703_002.tar.gz",
    "Handle_Plates_20250619_001.tar.gz",
    "Handle_The_Wooden_Barrel_Lid20250619_001.tar.gz",
    "Hang_Clothes20250625_002.tar.gz",
    "Hang_The_Towel_On_The_Shelf_20250715_006.tar.gz",
    "Hang_The_Towel_On_The_Shelf_20250722_008.tar.gz",
    "Hang_The_Towel_On_The_Shelf_20250805_012.tar.gz",
    "Heat_Food_In_Microwave_Oven_20250703_003.tar.gz",
    "Heat_Food_In_The_Microwave_20250710_006.tar.gz",
    "Insert_Flowers_Into_A_Vase_20250711_006.tar.gz",
    "Iron_The_Clothes_20250728_010.tar.gz",
    "Layered_Shoe_Cabinet_For_Placing_Slippers_20250728_010.tar.gz",
    "Lift_The_Steamer_Lid20250618_001.tar.gz",
    "Lift_up_the_carpet_and_wipe_the_floor_20250625_002.tar.gz",
    "Locker_torage_And_Retrieval20250623_002.tar.gz",
    "Make_Breakfast_20250626_002.tar.gz",
    "Make_Breakfast_20250726_010.tar.gz",
    "Make_Milk_Powder_20250728_010.tar.gz",
    "Make_Milk_Powder_20250731_012.tar.gz",
    "Make_Tea_20250624_002.tar.gz",
    "Make_The_Bed_20250616_001.tar.gz",
    "Make_The_Bed_20250708_006.tar.gz",
    "Make_The_Bed_20250730_012.tar.gz",
    "Milk_Powder_Preparation20250701_002.tar.gz",
    "Open_And_Close_Curtains_20250708_006.tar.gz",
    "Open_And_Close_Microwave_Oven_20250623_002.tar.gz",
    "Open_And_Close_Nightstand_Door_20250625_002.tar.gz",
    "Open_And_Close_Nightstand_Drawer_20250624_002.tar.gz",
    "Open_And_Close_The_Curtain_20250728_010.tar.gz",
    "Open_And_Close_The_Curtain_20250804_012.tar.gz",
    "Open_And_Close_The_Door_20250802_012.tar.gz",
    "Open_And_Close_The_Nightstand_To_Place_Items_20250802_012.tar.gz",
    "Open_And_Close_The_Notebook20250628_002.tar.gz",
    "Open_And_Close_The_Refrigerator_Drawer_To_Place_Items_20250728_010.tar.gz",
    "Open_And_Close_The_Refrigerator_Drawer_To_Place_Items_20250802_012.tar.gz",
    "Open_Close_Lid_Stirring_20250618_001.tar.gz",
    "Open_Or_Close_The_Cabinet_Door_To_Place_Items_20250719_008.tar.gz",
    "Open_Or_Close_The_Cabinet_Door_To_Place_Items_20250801_012.tar.gz",
    "Open_The_Bedroom_Door_To_Enter20250701_002.tar.gz",
    "Open_and_close_the_freezer_door_20250620.tar.gz",
    "Opening_Or_Closing_Refrigerator_Drawers_To_Store_And_Retrieve_Items20250624_002.tar.gz",
    "Organize_Dried_Fruit_And_Meat_Shelves_20250702_003.tar.gz",
    "Organize_Mineral_Water_Bottles_0n_The_Table_20250709_006.tar.gz",
    "Organize_Potato_Chip_Shelves_20250701_003.tar.gz",
    "Organize_Refrigerated_Beverage_Case_20250704_003.tar.gz",
    "Organize_Refrigerated_shelf_20250703_003.tar.gz",
    "Organize_Refrigerator_Items0250703_002.tar.gz",
    "Organize_The_Beverage_Cooler_20250630_003.tar.gz",
    "Organize_The_Bread_Shelf_20250630_003.tar.gz",
    "Organize_The_Coat_Rack20250617_001.tar.gz",
    "Organize_The_Condiment_Shelf_20250731_012.tar.gz",
    "Organize_The_Potato_Chip_Shelf_20250630_003.tar.gz",
    "Organize_The_Seasoning_Shelf_20250620_001.tar.gz",
    "Organize_The_Seasoning_Shelf_20250717_006.tar.gz",
    "Organize_The_Signs20250618_001.tar.gz",
    "Organize_The_Toiletries_20250719_008.tar.gz",
    "Organize_The_Wall-mounted_Coat_Rack_20250716_006.tar.gz",
    "Organize_Toys_20250628_002.tar.gz",
    "Organize_Toys_20250716_006.tar.gz",
    "Organize_trays_20250621_001.tar.gz",
    "Organizing_Desktop_Trash_20250630_003.tar.gz",
    "Pack_Medicines_Into_Medicine_Box_20250707_005.tar.gz",
    "Pen_Case_For_Storing_Stationery_20250717_006.tar.gz",
    "Pen_Case_For_Storing_Stationery_20250719_008.tar.gz",
    "Pen_Case_For_Storing_Stationery_20250802_012.tar.gz",
    "Pick_And_Place_Items_On_The_Bread_Shelf_20250701_003.tar.gz",
    "Pick_And_Place_Items_On_The_Potato_Chip_Shelves_20250701_003.tar.gz",
    "Pick_And_place_Ice_Cream_Freezer.tar.gz",
    "Pick_Up_Garbage_On_The_Ground_20250724_010.tar.gz",
    "Pick_Up_Garbage_On_The_Ground_20250801_012.tar.gz",
    "Pick_Up_Garbage_On_The_Ground_20250805_011.tar.gz",
    "Pick_Up_Or_Put_Down_The_Clip20250619.tar.gz",
    "Pick_Up_Trash_From_Ground_20250709_006.tar.gz",
    "Place_Slippers_In_The_Layered_Shoe_Cabinet_20250805_012.tar.gz",
    "Place_The_Dress_Shirt_On_The_Hanger20250619_001.tar.gz",
    "Plug_And_Unplug_Switch_Lights_20250726_010.tar.gz",
    "Plug_And_Unplug_Switch_Lights_20250802_012.tar.gz",
    "Plug_Into_A_Fixed_Socket_20250710_006.tar.gz",
    "Plug_Into_A_Fixed_Socket_20250722_008.tar.gz",
    "Plug_The_Socket_20250616_001.tar.gz",
    "PourWater_20250614.tar.gz",
    "Pour_Leftovers_20250802_012.tar.gz",
    "Pour_Out_The_Leftover_Food_20250714_006.tar.gz",
    "Pour_The_Leftover_Food_20250724_010.tar.gz",
    "Pour_Water20250708_006.tar.gz",
    "Pour_Water_20250801_012.tar.gz",
    "Pour_Water_20250806_011.tar.gz",
    "Pull_Out_The_Cap_20250725_010.tar.gz",
    "Push_And_Pull_The_Chair_20250728_010.tar.gz",
    "Push_In_Chairs_20250614_001.tar.gz",
    "Pushing_Garbage_Trucks_To_Put_Garbage_Away20250707_005.tar.gz",
    "Put_A_Bag_In_The_Trash_Can20250617_001.tar.gz",
    "Put_Clothes_Into_The_Basket_20250709_006.tar.gz",
    "Put_Clothes_Into_The_Basket_20250731_012.tar.gz",
    "Put_Clothes_Into_The_Basket_20250813_011.tar.gz",
    "Put_Clothes_Into_Wardrobe_20250707_005.tar.gz",
    "Put_Shoes_Into_The_Shoe_Box_20250801_012.tar.gz",
    "Put_Slippers_Into_Floor_Standing_Shoe_Cabinet_20250620_001.tar.gz",
    "Put_The_Flowers_In_The_Vase20250618_001.tar.gz",
    "Put_The_Glasses_In_The_Glasses_Case_20250722_008.tar.gz",
    "Put_The_Glasses_In_The_Glasses_Case_20250731_012.tar.gz",
    "Put_The_Items_Into_The_Storage_Box_20250929_002_007.tar.gz",
    "Put_The_Landline_Phone_Back_In_Place_20250725_010.tar.gz",
    "Put_The_Pen_Into_The_Pen_Holder_20250722_008.tar.gz",
    "Put_The_Pen_Into_The_Pen_Holder_20250724_010.tar.gz",
    "Put_The_Pen_Into_The_Pen_Holder_20250731_012.tar.gz",
    "Put_The_Phone_Back_In_place_20250716_006.tar.gz",
    "Put_The_Pillow_On_The_Bed20250617_001.tar.gz",
    "Put_The_Shoes_Into_The_Shoe_Box_202507116_006.tar.gz",
    "Put_The_Shoes_Into_The_Shoebox_20250725_010.tar.gz",
    "Put_The_Trash_Bag_At_The_Door20250626_002.tar.gz",
    "Put_the_shoes_into_the_shoe_box_20250621_001.tar.gz",
    "ReplaceFacialTissue_20250613_001.tar.gz",
    "Replace_The_Toilet_Paper_Roll_20250712_006.tar.gz",
    "Replace_The_Toilet_Paper_Roll_20250805_012.tar.gz",
    "Replace_The_Toiletries_20250616_001.tar.gz",
    "Replace_Toilet_Paper_20250623_002.tar.gz",
    "Retrieving_And_Placing_Items_Inside_The_TV_Cabinet_20250726_010.tar.gz",
    "Retrieving_And_Placing_Items_Inside_The_TV_Cabinet_20250805_012.tar.gz",
    "ReturnClothesToBasket_20250613_001.tar.gz",
    "Ring_The_Landline20250627_002.tar.gz",
    "Serve_Soup_In_The_Rice_Pot_20250717_006.tar.gz",
    "Serve_Soup_With_A_Rice_Pot_20250801_012.tar.gz",
    "Sliding_Bar_Stool_20250625_002.tar.gz",
    "Sort_Out_Desktop_Garbage20250802_012.tar.gz",
    "Steam_Rice_20250711_006.tar.gz",
    "Steam_Rice_20250721_008.tar.gz",
    "Steam_Rice_20250729_010.tar.gz",
    "Steam_Rice_20250730_012.tar.gz",
    "Stir-Fry_Dishes_20250726_010.tar.gz",
    "Stir-fry_20250722_008.tar.gz",
    "Stir-frying_20250627_002.tar.gz",
    "Storage_Tools_20250722_008.tar.gz",
    "Storage_Tools_20250725_010.tar.gz",
    "Storage_Tools_20250802_012.tar.gz",
    "Storage_Towels_20250715_006.tar.gz",
    "Storage_Towels_20250721_008.tar.gz",
    "Store_Dolls_20250812_011.tar.gz",
    "Store_Fruit_20250811_011.tar.gz",
    "Store_Tableware_20250625_002.tar.gz",
    "Store_The_Dolls_20250719_008.tar.gz",
    "Switch_The_Fan_On_And_Off_20250725_010.tar.gz",
    "Switch_The_Fan_On_And_Off_20250804_012.tar.gz",
    "Switch_The_Rotating_Nightstand_And_Place_Items_20250804_012.tar.gz",
    "Switch_The_Sign_20250616_001.tar.gz",
    "Take_A_Drink_Out_Of_Fridge_And_Put_It_On_Table_20250712_006.tar.gz",
    "Take_And_Place_The_Portable_Power_Bank20250628_002.tar.gz",
    "Take_And_Put_Items_In_And_Out_Of_The_Microwave_Oven_20250731_012.tar.gz",
    "Take_Drinks_From_The_Refrigerator_And_Place_Them_On_The_Table_20250728_010.tar.gz",
    "Take_Out_The_Laundry_From_The_Washing_Machine20250618_001.tar.gz",
    "Take_a_bowl_from_the_cabinet_20250620_001.tar.gz",
    "Take_and_place_tableware_20250620_001.tar.gz",
    "Taking_And_Placing_Shoes_In_The_Shoe_Cabinet20250624_002.tar.gz",
    "Tidy_Up_Bathroom_Toiletries_20250623_002.tar.gz",
    "Toast_Bread_20250618_001.tar.gz",
    "Turn_0n_The_Food_Pot20250618_001.tar.gz",
    "Turn_On_And_Off_Central_Air_Conditioning_System_20250623_002.tar.gz",
    "Turn_On_And_Off_Central_Air_Conditioning_System_20250624_002_special.tar.gz",
    "Turn_On_And_Off_The_Air_Conditioner_20250802_012.tar.gz",
    "Turn_On_And_Off_The_Small_Fan_20250711_006.tar.gz",
    "Turn_On_And_Turn_Off_The_Air_Conditioner_20250724_010.tar.gz",
    "Turn_On_Off_The_Light_20250619_001.tar.gz",
    "Twist_The_Bottle_Cap20250619_001.tar.gz",
    "Twist_The_Bottle_Cap_20250725_010.tar.gz",
    "Twist_The_Bottle_Cap_20250731_012.tar.gz",
    "Uncap_The_Pen_20250804_012.tar.gz",
    "Use_Of_Objects_20250704_003.tar.gz",
    "Use_The_Drawer_To_Store_The_Hair_Dryer20250625_002.tar.gz",
    "Washing_Machine_Washes_Clothes_20250728_010.tar.gz",
    "Water_Dispenser_Connected_To_Water20250707_005.tar.gz",
    "Wearing_A_Strip_Rack_Hanging_Clothes_For_Retrieval_20250724_010.tar.gz",
    "Wipe_The_Desktop_20250801_012.tar.gz",
    "Wipe_The_Desktop_20250804_011.tar.gz",
    "Wipe_The_Sewage_Stains_With_A_Ground_Cloth_20250709_006.tar.gz",
    "Wipe_The_Sewage_Stains_With_A_Ground_Cloth_20250725_010.tar.gz",
    "Wipe_The_Sewage_Stains_With_A_Ground_Cloth_20250801_012.tar.gz",
)
"""Every tarball under ``lerobot/`` at ``REVISION``, in build order."""
SCHEMA_SKIPPED = (
    "Clean_The_Sofa20250627_002.tar.gz",
    "Organize_The_Beverage_Cooler_20250630_003.tar.gz",
    "Organize_The_Bread_Shelf_20250630_003.tar.gz",
    "Put_The_Items_Into_The_Storage_Box_20250929_002_007.tar.gz",
)
"""The tarballs whose schema differs from the layouts (7-dim arms)."""
SUBSET = (
    "Organize_Toys_20250628_002.tar.gz",
    "Put_The_Items_Into_The_Storage_Box_20250929_002_007.tar.gz",
)
"""35 episodes and one schema-skipped tarball."""
FPS_SKIPPED = {
    "Boil_The_Water_20250714_006.tar.gz": (87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 101, 102)
    + (104, 106, 107, 108),
}
"""Upstream episodes (after the quality filter) whose videos and parquet run at 62 fps: one
recording session of robot RB250319004 (HF discussion #10). They are left out (17 episodes,
89,471 frames)."""

QUALITY_EXCLUDE = {"unqualified", "null"}
STATE_LAYOUT = (
    ("observation.state.left_arm", 6),
    ("observation.state.left_arm.velocities", 6),
    ("observation.state.right_arm", 6),
    ("observation.state.right_arm.velocities", 6),
    ("observation.state.torso", 4),
    ("observation.state.torso.velocities", 4),
    ("observation.state.chassis.imu", 10),
    ("observation.state.chassis", 3),
    ("observation.state.chassis.velocities", 3),
    ("observation.state.left_gripper", 1),
    ("observation.state.right_gripper", 1),
    ("observation.state.left_ee_pose", 7),
    ("observation.state.right_ee_pose", 7),
)
ACTION_LAYOUT = (
    ("action.left_arm", 6),
    ("action.right_arm", 6),
    ("action.left_gripper", 1),
    ("action.right_gripper", 1),
    ("action.torso.velocities", 6),
    ("action.chassis.velocities", 6),
)


def source(tarballs: tuple[str, ...]) -> HfSnapshot:
    return HfSnapshot(REPO, REVISION, tuple(f"lerobot/{t}" for t in tarballs))


PINS = {
    NAME: Pin(
        (source(TARBALLS),),
        episodes=16552,
        frames=18672026,
        subset=Pin((source(SUBSET),), episodes=35, frames=88710),
    ),
}


def english(text: str) -> str:
    """The English side of ``中文@English`` (other strings stripped)."""
    return text.partition("@")[2].strip() if "@" in text else text.strip()


def schema_ok(info: dict) -> bool:
    features = info["features"]
    return all(
        name in features and features[name]["shape"] == [dim]
        for name, dim in (*STATE_LAYOUT, *ACTION_LAYOUT)
    )


def concat(path: Path, layout: tuple[tuple[str, int], ...]) -> np.ndarray:
    """The ``layout`` columns of one episode parquet side by side, float32."""
    return v30.concat_columns(v30.read_columns(path, [name for name, _ in layout]), layout)[0]


def runs(texts: list[str]) -> list[dict]:
    """``{instruction, start, end}`` per run of equal consecutive texts."""
    return [{"instruction": t, "start": a, "end": b} for a, b, t in lookup.runs(texts)]


def video_fps(path: Path) -> float:
    num, den = ffprobe(path, "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0").split("/")
    return float(num) / float(den)


def episodes_of(
    src: Path, rules: TextRules, off_rate: list[int]
) -> Iterator[tuple[v30.Episode, list[dict]]]:
    """The kept episodes of one extracted tarball and their instruction segments; the episodes
    left out for a video rate other than ``FPS`` go to ``off_rate``."""
    info = json.loads((src / "meta" / "info.json").read_text())
    tasks = {t["task_index"]: t["task"] for t in v30.read_jsonl(src / "meta" / "tasks.jsonl")}
    quality = {i for i, t in tasks.items() if t in QUALITY_EXCLUDE}
    for meta in v30.read_jsonl(src / "meta" / "episodes.jsonl"):
        ep = meta["episode_index"]
        path, videos = v30.v2x_files(src, info, ep)
        cols = v30.read_columns(
            path, ["task_index", "coarse_task_index", "quality_index", "coarse_quality_index"]
        )
        if len(cols["task_index"]) != meta["length"]:
            raise ValueError(f"{path}: {len(cols['task_index'])} rows, meta says {meta['length']}")
        if quality & {int(q) for c in ("quality_index", "coarse_quality_index") for q in cols[c]}:
            continue
        coarse = english(tasks.get(int(cols["coarse_task_index"][0]), ""))
        if not coarse or coarse in QUALITY_EXCLUDE:
            continue
        fine = []
        for t in cols["task_index"]:
            text = english(tasks.get(int(t), ""))
            fine.append(coarse if not text or text in QUALITY_EXCLUDE else text)
        rates = {video_fps(video) for video in videos.values()}
        if rates != {FPS}:
            logger.info(
                "%s episode %d: left out, video at %s fps, not %d", src.name, ep, rates, FPS
            )
            off_rate.append(ep)
            continue
        yield (
            v30.Episode(
                ep,
                concat(path, STATE_LAYOUT),
                concat(path, ACTION_LAYOUT),
                rules.task(coarse),
                video_src=videos,
                frame_tasks=[rules.task(t) for t in fine],
            ),
            runs(fine),
        )


class Galaxea(lookup.TarballLeaf):
    leaf, fps, robot_type = LEAF, FPS, ROBOT_TYPE

    def __init__(self) -> None:
        self.skipped: list[str] = []
        self.off_rate: dict[str, list[int]] = {}

    def tarball_episodes(
        self, name: str, archive: Path, work: Path, rules: TextRules
    ) -> Iterator[tuple[v30.Episode, list[dict]]]:
        """Extracted into ``work`` (the videos are stream-copied from it); a tarball outside
        the layouts gives none."""
        untar(archive, work, fresh=True)
        data = work / archive.name.removesuffix(".tar.gz")
        if not schema_ok(json.loads((data / "meta" / "info.json").read_text())):
            self.skipped.append(archive.name)
            logger.info("%s: %s skipped (schema: 7-dim arms)", name, archive.name)
            return
        n = 0
        off_rate: list[int] = []
        for n, item in enumerate(episodes_of(data, rules, off_rate), 1):
            yield item
        if off_rate:
            self.off_rate[archive.name] = off_rate
        logger.info("%s: %s, %d episodes", name, archive.name, n)

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        self.skipped.clear()
        self.off_rate.clear()
        super().build(name, root, pin, shard)
        (src,) = pin.sources
        assert isinstance(src, HfSnapshot) and src.allow is not None
        expected = [t for t in SCHEMA_SKIPPED if f"lerobot/{t}" in src.allow]
        if self.skipped != expected:
            raise ValueError(f"{name}: schema-skipped {self.skipped}, pinned {expected}")
        rates = {t: list(e) for t, e in FPS_SKIPPED.items() if f"lerobot/{t}" in src.allow}
        if self.off_rate != rates:
            raise ValueError(f"{name}: left out for the video rate {self.off_rate}, pinned {rates}")


RECIPES = {NAME: Recipe(Galaxea(), curate=("frozen",), stats=(op("kept_qk"),))}

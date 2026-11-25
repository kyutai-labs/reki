"""Reader for the RDT-1B fine-tune data (``rdt-ft-data``: AgileX Cobot Magic, bimanual): one
leaf per task directory, a collection under ``rdt_lerobot/``.

Upstream is one tar.gz split into 81 parts (``rdt_data.tar.gz.NN``), read as one stream (cat
then untar, in the download stage) into ``rdt_data/<task>/episode_N.hdf5`` (+ ``expanded_instruction_gpt-4-turbo.json``).
Per task: episodes in sorted file-name order; an episode shorter than 2 frames or with a NaN in
its state/action is left out (4 of 6,110). Per episode: state = ``observations/qpos`` (14),
action = ``action`` (14), the three JPEG camera streams decoded and encoded (``v30.ENCODER``).
Task text: the instruction json's ``instruction``, else the directory name.

``--subset``: only the first part. Its stream ends inside a task; the tasks it holds whole
(every member before the next task starts) are built.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common import v30
from ..common.download import ConcatFile, umask_filter, upstream_dir
from ..common.stats import op
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, Recipe, collection_dir

logger = logging.getLogger(__name__)

NAME = "rdt"
REPO = "robotics-diffusion-transformer/rdt-ft-data"
REVISION = "a3f4b6624d4fc3a09b538f65081b37401fd8bd84"
PARTS = tuple(f"rdt_data.tar.gz.{i:02d}" for i in range(81))
FPS = 25
ROBOT_TYPE = "aloha"
CAMERAS = {"cam_high": "high", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
INSTRUCTION = "expanded_instruction_gpt-4-turbo.json"

TASKS = (
    "airpods_on_second_layer",
    "airpods_on_third_layer",
    "arrange_fruits_by_size",
    "arrange_word_2024",
    "arrange_word_23",
    "arrange_word_ABCD",
    "arrange_word_ALOHA",
    "arrange_word_DEEP",
    "arrange_word_LEARNING",
    "arrange_word_THU",
    "arrange_word_TURING",
    "choose_toy_by_color",
    "choose_toy_by_size",
    "clean_pour_water",
    "close_glasses_box",
    "close_laptop",
    "close_laptop_2",
    "close_the_book",
    "cocktail_sunset_coconut",
    "cocktail_sunset_ice",
    "cocktail_sunset_lemon",
    "cocktail_sunset_orange",
    "cocktail_sunset_pineapple",
    "cocktail_sunset_red",
    "coffee_cup_left_PC",
    "coffee_cup_left_to_left",
    "coffee_cup_middle_to_left",
    "coffee_cup_middle_to_right",
    "coffee_cup_on_PC",
    "coffee_cup_right_PC",
    "coffee_cup_right_to_right",
    "collect_earphone",
    "collect_litter",
    "collect_pen",
    "connect_charging_cable",
    "cover_laptop",
    "cover_mirror_jsh",
    "cover_spary_lid",
    "draw_char_A",
    "draw_char_Z",
    "draw_check_mark",
    "draw_cross",
    "draw_line",
    "draw_rectangle",
    "draw_triangle",
    "exchange_cup_position",
    "flip_bottle",
    "flip_calendar_page",
    "flowering_fake",
    "flowering_fake_2",
    "fold_towel",
    "gather_paper_ball",
    "get_cold_water",
    "get_hot_water",
    "grab_stick_into_bottle",
    "handover_pan",
    "hang_bag_on_hook",
    "headphone_left_phone",
    "hook_earring",
    "hook_keys",
    "insert_blue_shoe",
    "insert_book_shelf",
    "insert_cable_charger",
    "insert_cube_slot",
    "insert_pen_container",
    "insert_phone_2",
    "insert_phone_charger",
    "kiwi_toward_upside_down_mouse",
    "match_the_counter",
    "math_cube_with_fruit_jsh",
    "math_cube_with_rag_jsh",
    "messy_pour_water",
    "messy_stack_tomatoes",
    "move_foam_mahjong_from_right_to_left",
    "move_object_near_mirror",
    "open_laptop",
    "pack_durant_shirt",
    "pack_nvidia_shirt",
    "pack_pants",
    "pack_soccer_shirt",
    "paper_on_second_layer",
    "paper_on_third_layer",
    "pick_apple_into_bag",
    "pick_backpack_from_chair_to_desk",
    "pick_badminton",
    "pick_book_into_backpack",
    "pick_bottle_get_water",
    "pick_chips_to_box",
    "pick_different_doll",
    "pick_key_to_basket",
    "pick_kiwifruit_into_bag",
    "pick_larger_value",
    "pick_magiccube_into_bag",
    "pick_marker_pen_from_cup",
    "pick_mask",
    "pick_orange_into_bag",
    "pick_out_clip",
    "pick_pen",
    "pick_pen_on_notebook",
    "pick_place_pineapple",
    "pick_place_ver_water_bottle",
    "pick_place_water_bottle",
    "pick_potato_chip_packaging_near_apple",
    "pick_power_bank",
    "pick_same_color_clip",
    "pick_staples_into_stapler",
    "pick_tape_into_bag",
    "pick_the_different_dice",
    "pick_tomato_to_desk",
    "pick_up_letter_c",
    "pick_up_the_pen",
    "pin_into_second_layer",
    "place_cube_in_the_center",
    "place_cube_in_the_center_2",
    "place_marker",
    "place_object_on_mirror",
    "place_octopus_upright",
    "place_phone",
    "playmahjong",
    "plug_charger",
    "pour_clip_paper_box",
    "pour_water_4",
    "pour_water_5",
    "pour_water_6",
    "pour_water_between_cup",
    "pour_water_bottle2cup_clean",
    "pour_water_bottle2cup_mess",
    "pour_water_can2cup",
    "pour_water_cup2jigger2glass",
    "pour_water_cup_changing_light",
    "pour_water_cup_full",
    "pour_water_cup_half",
    "pour_water_cup_little",
    "pour_water_cup_scene1",
    "pour_water_cup_scene2",
    "pour_water_cup_yellow_light",
    "pour_water_dark_2",
    "pour_water_dark_meeting_room",
    "pour_water_from_bottle_to_cup",
    "pour_water_g2k",
    "pour_water_glass_changing_light",
    "pour_water_jigger2cup",
    "pour_water_left_hand",
    "pour_water_meeting_room",
    "press_alcohol_sanitizing",
    "press_socket_button",
    "press_stapler",
    "prop_bottle",
    "pull_chair",
    "pull_paper_on_id_card_jsh",
    "pull_trash_can",
    "pull_wet_wipe",
    "push_add_chip",
    "push_chair",
    "push_container_back_to_chips_packaging",
    "push_max_chip",
    "push_min_chip",
    "push_mirror",
    "push_mirror_surface",
    "put_badminton_in_order",
    "put_cherry_bowl",
    "put_clothes_into_backpack",
    "put_cup_behind_laptop",
    "put_cup_front_laptop",
    "put_cup_left_laptop",
    "put_cup_right_laptop",
    "put_doll_on_coin",
    "put_fluffy_octopus_into_bowl",
    "put_glasses_into_box",
    "put_glue_into_box_yellow_light",
    "put_ice_scoop_in_box_messy_table",
    "put_ice_scoop_in_box_tidy_table",
    "put_kiwi_on_doll",
    "put_kiwifruit_into_box_yellow_light",
    "put_marker_pen_in_cup",
    "put_mouse_on_pad",
    "put_object_into_cabinet",
    "put_object_into_drawer",
    "put_orange_into_box",
    "put_orange_into_microwave",
    "put_orange_paperbox",
    "put_paper_cup_jsh",
    "put_paper_upside_down",
    "put_rag_on_laptop_blue",
    "put_rag_on_laptop_orange",
    "put_rag_on_laptop_purple",
    "put_rag_on_laptop_yellow",
    "put_rubbish_in_carton",
    "put_snack_into_microwave",
    "put_spitballs_in_carton",
    "put_spitballs_in_carton_2",
    "put_sponge_in_box",
    "put_sponge_into_drawer",
    "putmajiangintobox",
    "redirect_magiccube",
    "robot_dog_backward",
    "robot_dog_forward",
    "robot_dog_forward_2",
    "roll_dice",
    "roll_dice_and_compare",
    "search_wipe_glass_water",
    "shake_glass",
    "shovel_ice_into_cup",
    "sort_fruits",
    "sort_toy_by_color",
    "sort_toy_by_size",
    "sortmahjong",
    "spell_big",
    "spell_car",
    "spell_cool",
    "spell_cup",
    "spell_dog",
    "spell_love",
    "spell_sky",
    "spell_sun",
    "spell_yes",
    "stack_can",
    "stack_cap_on_apple_yellow_light",
    "stack_cup",
    "stack_letter_block_easy",
    "stack_letter_brick",
    "stack_letter_cube_2",
    "stack_magiccube_on_glass_yellow_light",
    "stack_tomato_cans",
    "stack_tomatoes_2",
    "stack_tomatoes_3",
    "stack_tomatoes_meeting_room",
    "stir_liquid_inside_cup",
    "straw_cafe_cup_l",
    "straw_cafe_cup_m",
    "straw_cafe_cup_s",
    "stuckmahjong",
    "swipe_the_letter",
    "take_clothes_into_backpack",
    "take_off_lid_of_ice_box",
    "take_out_chips_from_packaging",
    "take_out_glassescase_from_bag",
    "take_out_pen_lid",
    "take_out_spary_lid",
    "take_out_tissue",
    "take_sponge_from_box",
    "takemahjong",
    "takeout_insert_usb",
    "tape_on_second_layer",
    "tape_on_third_layer",
    "tian_fold_cloth",
    "tian_open_box_pick_bottle",
    "tian_open_laptop_press_keyboard",
    "tian_put_red_book_top",
    "turn_off_light",
    "turn_on_light",
    "turn_over_foam_mahjong",
    "unplug_charger",
    "unplug_data_cable",
    "unscrew_cap",
    "unwind_charging_cable",
    "unzip_the_bag",
    "up_down_cube",
    "wash_cup_1",
    "wash_cup_2",
    "wash_cup_3",
    "wash_cup_5",
    "wash_cup_6",
    "wash_water",
    "water_plant_1",
    "water_plant_2",
    "water_plant_4",
    "wipe_cherry",
    "wipe_desk",
    "wipe_glass_water",
    "wipe_laptop_water",
    "wipe_table_b2f",
    "wipe_table_f2b",
    "wipe_table_l2r",
    "wipe_table_l2r_avoid",
    "wipe_table_r2l",
    "wipe_table_r2l_avoid",
    "wipe_whitboard_wdy",
    "wipe_whiteboard",
    "write_board_0",
    "write_board_1",
    "write_board_1+1",
    "write_board_10",
    "write_board_11",
    "write_board_12",
    "write_board_13",
    "write_board_14",
    "write_board_15",
    "write_board_16",
    "write_board_17",
    "write_board_18",
    "write_board_19",
    "write_board_2",
    "write_board_20",
    "write_board_3",
    "write_board_4",
    "write_board_5",
    "write_board_6",
    "write_board_7",
    "write_board_8",
    "write_board_9",
    "write_hi_whiteboard",
    "zip_the_bag",
)
"""Every task directory (= leaf), sorted."""
SUBSET_TASKS = ("close_glasses_box", "pick_staples_into_stapler")
"""Two of the tasks the first part holds whole."""


def source(parts: tuple[str, ...]) -> HfSnapshot:
    return HfSnapshot(REPO, REVISION, parts)


PINS = {
    NAME: Pin(
        (source(PARTS),),
        episodes=6106,
        frames=3205122,
        subset=Pin((source(PARTS[:1]),), episodes=40, frames=11384),
    ),
}


def _tasks(pin: Pin) -> tuple[str, ...]:
    (src,) = pin.sources
    assert isinstance(src, HfSnapshot)
    return TASKS if src.allow == PARTS else SUBSET_TASKS


def untar(parts: list[Path], out: Path, tasks: tuple[str, ...], whole: bool) -> None:
    """Extract the members of ``tasks`` from the split archive ``parts`` into ``out`` (once: a
    ``.done`` marker). ``whole``: every part is there and the stream must end cleanly; else the
    stream may end early, and each wanted task must have ended before it does."""
    done = out / ".done"
    if done.exists():
        return
    wanted = set(tasks)
    finished: set[str] = set()
    current = None
    stream = io.BufferedReader(ConcatFile(parts), buffer_size=1 << 24)
    member_filter = umask_filter()
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for member in tf:
                task = Path(member.name).parts[1] if len(Path(member.name).parts) > 1 else None
                if task != current:
                    if current is not None:
                        finished.add(current)
                    current = task
                if task in wanted:
                    tf.extract(member, out, filter=member_filter)
                if not whole and wanted <= finished:
                    break
    except (EOFError, tarfile.ReadError):
        if whole:
            raise
    if current is not None and whole:
        finished.add(current)
    missing = wanted - finished
    if missing:
        raise ValueError(f"{parts[0].name}…: tasks not whole in the archive: {sorted(missing)}")
    done.touch()


def task_text(task_dir: Path) -> str:
    path = task_dir / INSTRUCTION
    if path.is_file():
        instruction = json.loads(path.read_text()).get("instruction")
        if instruction:
            return instruction.strip()
    return task_dir.name.replace("_", " ").strip().capitalize() + "."


def _decode(jpeg: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)


def task_episodes(task_dir: Path) -> Iterator[v30.Episode]:
    import h5py  # type: ignore[import-not-found]

    task = task_text(task_dir)
    for i, path in enumerate(sorted(task_dir.glob("episode_*.hdf5"))):
        with h5py.File(path, "r") as f:
            state = np.asarray(f["observations/qpos"], dtype=np.float32)
            action = np.asarray(f["action"], dtype=np.float32)
            if len(state) < 2 or np.isnan(state).any() or np.isnan(action).any():
                logger.info(
                    "%s: %s left out (length %d / NaN)", task_dir.name, path.name, len(state)
                )
                continue
            frames = {
                f"observation.images.{name}": np.stack(
                    [_decode(b) for b in f[f"observations/images/{raw}"][:]]
                )
                for raw, name in CAMERAS.items()
            }
        yield v30.Episode(i, state, action, task, frames=frames)


@dataclass(frozen=True)
class Rdt(Reader):
    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [collection_dir(name, root) / t for t in _tasks(pin)]

    def download(self, name: str, root: Path, pin: Pin) -> None:
        """The parts, then their tasks untarred into ``<upstream>/extracted``."""
        super().download(name, root, pin)
        (src,) = pin.sources
        assert isinstance(src, HfSnapshot) and src.allow is not None
        up = upstream_dir(root, src)
        untar([up / p for p in src.allow], up / "extracted", _tasks(pin), src.allow == PARTS)

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        (src,) = pin.sources
        raw = upstream_dir(root, src) / "extracted"
        if not (raw / ".done").exists():
            raise FileNotFoundError(f"{raw}: not extracted (run the download stage)")
        tasks = _tasks(pin)
        i, n = shard
        for task, out in list(zip(tasks, self.leaves(name, root, pin)))[i::n]:
            v30.write_leaf(task_episodes(raw / "rdt_data" / task), out, FPS, ROBOT_TYPE)
            logger.info("%s: %s built", name, task)


RECIPES = {
    NAME: Recipe(Rdt(), curate=("frozen",), stats=(op("global_qk"),)),
}

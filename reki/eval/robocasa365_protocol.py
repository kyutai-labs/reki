"""RoboCasa365 official eval protocol (multitask track).

Training uses Human300, the 300 pretraining tasks in the registered ``robocasa365`` collection.
Evaluation runs 50 binary-success rollouts for each of the 50 target tasks in the pretraining
kitchen scenes (``split="pretrain"``). Composite-Unseen tasks are not in Human300 (zero-shot,
no dataset leaf) and use collection-wide statistics. The suite lists come from
``robocasa.utils.dataset_registry.TARGET_TASKS`` (v1.0.1), with 18, 16, and 16 tasks. The score
is the macro mean across all 50 tasks.
"""

EVAL_SPLIT = "pretrain"

TARGET_SUITES: dict[str, list[str]] = {
    "atomic_seen": [
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    ],
    "composite_seen": [
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "PackIdenticalLunches",
        "PreSoakPan",
        "PrepareCoffee",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "WashLettuce",
    ],
    "composite_unseen": [
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "GarnishPancake",
        "GatherTableware",
        "HeatKebabSandwich",
        "MakeIceLemonade",
        "PanTransfer",
        "PortionHotDogs",
        "RecycleBottlesByType",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    ],
}

TARGET50: list[str] = [t for tasks in TARGET_SUITES.values() for t in tasks]
SUITE_OF: dict[str, str] = {t: s for s, tasks in TARGET_SUITES.items() for t in tasks}

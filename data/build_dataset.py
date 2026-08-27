"""Builds the labeled ad-creative dataset from the Oxford-IIIT Pet images.

Each Oxford Pet image becomes one "ad creative" by pairing it with generated
ad copy for a pet product. We use real product photos with synthetic ad text
because no public dataset of real ads with policy-violation ground truth
exists -- see README for the full rationale.

Label logic (deliberately one axis of "wrong" per non-approved class, so the
labels aren't confounded and a classifier failure on one class is
diagnosable):

  approved            - species-correct product copy, no violation language,
                         full-resolution image.
  policy_violation     - species-correct copy, but the copy itself contains a
                         banned claim pattern (health claim, fake urgency,
                         superlative, fear-based language). Image untouched.
  misleading            - copy for the WRONG species (dog copy on a cat photo
                         or vice versa). Copy itself is otherwise clean.
                         Image untouched.
  low_quality           - species-correct, clean copy, but EITHER the image is
                         heavily degraded (blur + downsample/upsample) OR the
                         copy is spammy/incoherent. Roughly half and half.

Oxford Pet filename convention: breed is embedded in the filename, cats are
capitalized ("Siamese_12.jpg"), dogs are lowercase ("pug_52.jpg"). This is
the dataset's own convention, not something we're inferring.
"""
import random
import shutil
from pathlib import Path

from PIL import Image, ImageFilter

RAW_IMAGES_DIR = Path("data/raw/oxford-iiit-pet/images")
OUT_IMAGES_DIR = Path("data/images")
MANIFEST_PATH = Path("data/dataset.csv")

RNG_SEED = 42
N_PER_CLASS = 120  # 4 classes x 120 = 480 total

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
}

# ---- ad copy template pools -------------------------------------------

DOG_PRODUCTS = [
    "grain-free dog food", "joint & mobility support chew", "no-pull dog harness",
    "durable rubber chew toy", "oatmeal shampoo for sensitive skin",
    "grain-free training treat mix", "orthopedic dog bed", "slow-feed dog bowl",
]
CAT_PRODUCTS = [
    "grain-free cat food", "clumping cat litter", "interactive feather wand toy",
    "sisal scratching post", "hairball control treat mix", "soft-sided cat carrier",
    "self-grooming slicker brush", "ceramic cat feeding bowl",
]

APPROVED_TEMPLATES = [
    "Made for {breed}s and other {species}s, this {product} is a favorite with owners who want a reliable everyday option.",
    "Our {product} is formulated with {species} nutrition in mind -- a solid pick for {breed}s at any age.",
    "Owners of {breed}s reach for this {product} because it does the basics well, day after day.",
    "This {product} was designed with {species} comfort in mind and works well for breeds like the {breed}.",
    "A dependable {product} for {species} owners, tested with breeds including the {breed}.",
]

HEALTH_CLAIM_TEMPLATES = [
    "This {product} CURES joint pain in {species}s within days -- guaranteed or your money back.",
    "Clinically proven to eliminate {species} anxiety completely, this {product} reverses years of stress in one week.",
    "Vets agree: this {product} guarantees your {breed} will live 5 years longer. No other {product} comes close.",
    "Stop your {breed}'s suffering today -- this {product} is the only thing that actually works, period.",
]
URGENCY_CLAIM_TEMPLATES = [
    "Only 2 left in stock! Sale ends tonight -- get this {product} before it's gone forever.",
    "LAST CHANCE: this {product} sells out every time, buy now before the price doubles at midnight!",
]
SUPERLATIVE_CLAIM_TEMPLATES = [
    "The #1 vet-recommended {product} in America -- nothing else on the market even compares.",
    "Rated the best {product} in the world by every major review site, hands down.",
]

MISLEADING_TEMPLATES = APPROVED_TEMPLATES  # reused, but species is swapped at generation time

LOW_QUALITY_TEXT_TEMPLATES = [
    "{PRODUCT_UPPER} {PRODUCT_UPPER} BEST DEAL CLICK NOW SALE SALE BUY BUY {SPECIES_UPPER}",
    "product good buy now {product} {product} cheap price fast shipping click",
    "{product}!!!! !!! best!! {species} {product} now now now discount discount",
]

random.seed(RNG_SEED)


def species_and_breed(filename: str) -> tuple[str, str]:
    stem = filename.rsplit("_", 1)[0]  # strip trailing _<id>
    species = "cat" if stem[0].isupper() else "dog"
    breed = stem.replace("_", " ")
    return species, breed


def discover_breed_pools(files: list[Path]) -> tuple[list[str], list[str]]:
    """Scans actual filenames for the full breed vocabulary, split by species.
    Derived from the real files rather than hand-typed so it can't drift out
    of sync with what's actually on disk.
    """
    cat_breeds, dog_breeds = set(), set()
    for f in files:
        species, breed = species_and_breed(f.name)
        (cat_breeds if species == "cat" else dog_breeds).add(breed)
    return sorted(cat_breeds), sorted(dog_breeds)


def degrade_image(src_path: Path, dst_path: Path) -> None:
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    small = img.resize((max(1, w // 10), max(1, h // 10)), Image.BILINEAR)
    blurred = small.filter(ImageFilter.GaussianBlur(radius=2))
    degraded = blurred.resize((w, h), Image.BILINEAR)
    degraded.save(dst_path, quality=15)


def make_text(label: str, species: str, breed: str, cat_breeds: list[str], dog_breeds: list[str], subtype: str | None = None) -> str:
    products = DOG_PRODUCTS if species == "dog" else CAT_PRODUCTS
    product = random.choice(products)

    if label == "approved":
        tmpl = random.choice(APPROVED_TEMPLATES)
        return tmpl.format(breed=breed, species=species, product=product)

    if label == "policy_violation":
        pool = random.choice([HEALTH_CLAIM_TEMPLATES, URGENCY_CLAIM_TEMPLATES, SUPERLATIVE_CLAIM_TEMPLATES])
        tmpl = random.choice(pool)
        return tmpl.format(breed=breed, species=species, product=product)

    if label == "misleading":
        wrong_species = "cat" if species == "dog" else "dog"
        wrong_products = CAT_PRODUCTS if species == "dog" else DOG_PRODUCTS
        wrong_product = random.choice(wrong_products)
        tmpl = random.choice(MISLEADING_TEMPLATES)
        # describe a DIFFERENT breed of the wrong species so nothing in the text
        # matches the image -- sampled from the full breed pool, not a fixed
        # name, so the text can't be fingerprinted by a single constant string
        wrong_breed_pool = cat_breeds if wrong_species == "cat" else dog_breeds
        filler_breed = random.choice(wrong_breed_pool)
        return tmpl.format(breed=filler_breed, species=wrong_species, product=wrong_product)

    if label == "low_quality" and subtype == "text":
        tmpl = random.choice(LOW_QUALITY_TEXT_TEMPLATES)
        return tmpl.format(
            product=product, species=species,
            PRODUCT_UPPER=product.upper(), SPECIES_UPPER=species.upper(),
        )

    if label == "low_quality" and subtype == "image":
        tmpl = random.choice(APPROVED_TEMPLATES)
        return tmpl.format(breed=breed, species=species, product=product)

    raise ValueError(f"unhandled label/subtype: {label}/{subtype}")


def main() -> None:
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(RAW_IMAGES_DIR.glob("*.jpg"))
    print(f"found {len(all_files)} candidate jpg files")

    # dedupe by breed_id stem in case trainval+test overlap in filenames (they don't, but verify)
    usable = []
    skipped = 0
    for f in all_files:
        try:
            img = Image.open(f)
            img.verify()
        except Exception:
            skipped += 1
            continue
        usable.append(f)
    print(f"usable (opens cleanly): {len(usable)}, skipped (corrupt): {skipped}")

    cat_breeds, dog_breeds = discover_breed_pools(usable)
    print(f"discovered {len(cat_breeds)} cat breeds, {len(dog_breeds)} dog breeds")

    random.shuffle(usable)
    needed = N_PER_CLASS * 4
    if len(usable) < needed:
        raise RuntimeError(f"need {needed} usable images, only have {len(usable)}")

    pool = usable[:needed]
    labels = (
        ["approved"] * N_PER_CLASS
        + ["policy_violation"] * N_PER_CLASS
        + ["misleading"] * N_PER_CLASS
        + ["low_quality"] * N_PER_CLASS
    )
    assert len(pool) == len(labels)

    rows = []
    low_quality_counter = 0
    for src_path, label in zip(pool, labels):
        species, breed = species_and_breed(src_path.name)
        out_name = f"{len(rows):04d}_{src_path.name}"
        dst_path = OUT_IMAGES_DIR / out_name

        if label == "low_quality":
            subtype = "image" if low_quality_counter % 2 == 0 else "text"
            low_quality_counter += 1
            if subtype == "image":
                degrade_image(src_path, dst_path)
            else:
                shutil.copyfile(src_path, dst_path)
            text = make_text(label, species, breed, cat_breeds, dog_breeds, subtype=subtype)
        else:
            shutil.copyfile(src_path, dst_path)
            text = make_text(label, species, breed, cat_breeds, dog_breeds)

        rows.append({
            "image_path": str(dst_path).replace("\\", "/"),
            "ad_text": text,
            "ground_truth_label": label,
            "source_species": species,
            "source_breed": breed,
        })

    import csv
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {MANIFEST_PATH}")
    from collections import Counter
    print("label counts:", Counter(r["ground_truth_label"] for r in rows))
    print("species counts:", Counter(r["source_species"] for r in rows))


if __name__ == "__main__":
    main()

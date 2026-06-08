"""
Offline augmentacja klasy mniejszosciowej (zdrowe komorki - hem).

Zbior CNMC jest mocno niezbalansowany - w kazdym foldzie komorek chorych (all)
jest ~2x wiecej niz zdrowych (hem). Skutkuje to niska precyzja i recall na klasie
'hem' (model "ciagnie" predykcje w strone klasy wiekszosciowej).

Ten skrypt generuje dodatkowe, zaugmentowane kopie obrazow z folderow .../hem,
tak aby wyrownac liczebnosc klas (domyslnie hem:all = 1:1). Pliki zapisywane sa
w tym samym folderze hem z przyrostkiem '_augNNNN', dzieki czemu:
  * notebookowy `image_dataset_from_directory` automatycznie je zaladuje,
  * latwo je odroznic od oryginalow i usunac (--clean).

Obrazy CNMC to pojedyncze komorki wysegmentowane na CZARNYM tle (450x450),
dlatego transformacje geometryczne wypelniaja brzegi czernia (borderValue=0),
co idealnie zlewa sie z tlem.

Uzyte techniki augmentacji (komorki krwi nie maja wyrozniajacej orientacji,
wiec transformacje geometryczne sa w pelni naturalne):
  - losowe odbicia (poziome / pionowe),
  - losowa rotacja o dowolny kat 0-360 stopni,
  - losowe skalowanie (zoom in / out),
  - losowa translacja,
  - lagodny jitter kontrastu i gamma (multiplikatywny - czarne tlo zostaje czarne).

Wszystkie transformacje robione sa wylacznie na danych klasy 'hem' - klasy 'all'
nie ruszamy, bo to ona dominuje.

Przyklady uzycia:
    python utils/augment_hem.py                 # wyrownaj wszystkie foldy do 1:1
    python utils/augment_hem.py --ratio 1.0     # to samo, jawnie
    python utils/augment_hem.py --folds fold_0  # tylko jeden fold
    python utils/augment_hem.py --clean         # usun poprzednie augmentacje (potem dogeneruj)
    python utils/augment_hem.py --dry-run       # tylko pokaz plan, bez zapisu na dysk
"""

import argparse
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Konfiguracja sciezek / stalych
# ---------------------------------------------------------------------------
AUG_TAG = "_aug"  # przyrostek odrozniajacy pliki zaugmentowane od oryginalow
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DEFAULT_FOLDS = ["fold_0", "fold_1", "fold_2"]
IMG_EXTS = (".bmp", ".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Pomocnicze - listowanie plikow
# ---------------------------------------------------------------------------
def _is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def list_originals(folder):
    """Pliki oryginalne (bez przyrostka augmentacji), posortowane deterministycznie."""
    return sorted(
        f for f in os.listdir(folder)
        if _is_image(f) and AUG_TAG not in os.path.splitext(f)[0]
    )


def list_augmented(folder):
    """Pliki wczesniej zaugmentowane przez ten skrypt."""
    return [
        f for f in os.listdir(folder)
        if _is_image(f) and AUG_TAG in os.path.splitext(f)[0]
    ]


def count_images(folder):
    """Liczba wszystkich obrazow w folderze (oryginaly + augmentacje)."""
    return sum(1 for f in os.listdir(folder) if _is_image(f))


def next_aug_index(folder):
    """Kolejny wolny indeks augmentacji (kontynuacja, by uniknac kolizji nazw)."""
    mx = 0
    for f in list_augmented(folder):
        tail = os.path.splitext(f)[0].rsplit(AUG_TAG, 1)[-1]
        if tail.isdigit():
            mx = max(mx, int(tail))
    return mx + 1


# ---------------------------------------------------------------------------
# Augmentacja pojedynczego obrazu
# ---------------------------------------------------------------------------
def augment_image(
    img,
    rng,
    zoom=(0.85, 1.15),
    shift_frac=0.10,
    contrast=(0.85, 1.15),
    gamma=(0.85, 1.15),
):
    """
    Zwraca losowy wariant obrazu komorki.

    Kolejnosc: odbicia -> (rotacja + skala + translacja w jednej macierzy
    afinicznej, jedno przejscie interpolacji) -> lagodny jitter fotometryczny.
    Brzegi po transformacji geometrycznej wypelniane czernia (tlo preparatu).
    """
    h, w = img.shape[:2]

    # 1) losowe odbicia (komorka nie ma wyrozniajacej orientacji)
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)  # poziomo
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)  # pionowo

    # 2) rotacja (0-360) + skalowanie + translacja - jedna macierz afiniczna
    angle = rng.uniform(0.0, 360.0)
    scale = rng.uniform(*zoom)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += rng.uniform(-shift_frac, shift_frac) * w
    M[1, 2] += rng.uniform(-shift_frac, shift_frac) * h
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 3) lagodny jitter fotometryczny - multiplikatywny, wiec 0 (czern) zostaje 0
    alpha = rng.uniform(*contrast)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    g = rng.uniform(*gamma)
    if abs(g - 1.0) > 1e-3:
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / g)) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)

    return img


# ---------------------------------------------------------------------------
# Przetwarzanie jednego foldu
# ---------------------------------------------------------------------------
def process_fold(fold, ratio, rng, clean=False, dry_run=False):
    hem_dir = os.path.join(DATA_ROOT, fold, fold, "hem")
    all_dir = os.path.join(DATA_ROOT, fold, fold, "all")

    if not os.path.isdir(hem_dir) or not os.path.isdir(all_dir):
        print(f"[{fold}] POMINIETO - brak folderu hem/all ({hem_dir})")
        return 0

    # opcjonalne czyszczenie poprzednich augmentacji
    removed = 0
    if clean:
        for f in list_augmented(hem_dir):
            removed += 1
            if not dry_run:
                os.remove(os.path.join(hem_dir, f))

    originals = list_originals(hem_dir)
    n_orig = len(originals)
    n_all = count_images(all_dir)

    if n_orig == 0:
        print(f"[{fold}] POMINIETO - brak oryginalnych obrazow hem")
        return 0

    existing_aug = 0 if clean else len(list_augmented(hem_dir))
    total_now = n_orig + existing_aug
    target = int(round(n_all * ratio))
    to_generate = max(0, target - total_now)

    print(f"[{fold}] all={n_all}  hem_orig={n_orig}  hem_aug_istniejace={existing_aug}"
          + (f"  (usunieto={removed})" if clean else ""))
    print(f"[{fold}] cel hem={target} (ratio={ratio})  ->  do wygenerowania={to_generate}")

    if to_generate == 0:
        print(f"[{fold}] nic do zrobienia - klasa juz wyrownana\n")
        return 0

    idx = next_aug_index(hem_dir) if not clean else 1
    generated = 0
    i = 0
    while generated < to_generate:
        src = originals[i % n_orig]
        i += 1
        img = cv2.imread(os.path.join(hem_dir, src))
        if img is None:
            print(f"[{fold}] UWAGA: nie wczytano {src}, pomijam")
            continue
        aug = augment_image(img, rng)
        base, ext = os.path.splitext(src)
        out_name = f"{base}{AUG_TAG}{idx:04d}{ext}"
        if not dry_run:
            cv2.imwrite(os.path.join(hem_dir, out_name), aug)
        idx += 1
        generated += 1

    final_hem = total_now + generated
    print(f"[{fold}] {'(dry-run) ' if dry_run else ''}wygenerowano {generated} obrazow "
          f"-> hem={final_hem}, all={n_all} (ratio={final_hem / n_all:.2f})\n")
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Augmentacja zdrowych komorek (hem) w celu wyrownania klas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS,
                   help="Foldy do przetworzenia.")
    p.add_argument("--ratio", type=float, default=1.0,
                   help="Docelowy stosunek hem/all (1.0 = pelne wyrownanie).")
    p.add_argument("--clean", action="store_true",
                   help="Najpierw usun wczesniejsze augmentacje (pliki z '_aug').")
    p.add_argument("--dry-run", action="store_true",
                   help="Pokaz plan bez zapisu/usuwania plikow.")
    p.add_argument("--seed", type=int, default=42,
                   help="Ziarno generatora losowego (powtarzalnosc).")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print(f"Augmentacja klasy hem  |  ratio={args.ratio}  seed={args.seed}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print("=" * 70)

    total = 0
    for fold in args.folds:
        total += process_fold(fold, args.ratio, rng,
                              clean=args.clean, dry_run=args.dry_run)

    print("=" * 70)
    print(f"GOTOWE. Lacznie {'(dry-run) ' if args.dry_run else ''}wygenerowano {total} obrazow.")
    print("=" * 70)


if __name__ == "__main__":
    main()

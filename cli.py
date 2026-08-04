"""Командная строка: скрейп корпуса, сборка индекса, статус."""
from __future__ import annotations

import argparse
import sys

import db
import indexer
import llm
import scraper
from config import load_settings, load_sources

SETTINGS = load_settings()


def cmd_scrape(keys: list[str] | None) -> int:
    sources = load_sources()
    if keys:
        sources = [s for s in sources if s.key in keys]
        if not sources:
            print(f"Не найдено источников с ключами: {', '.join(keys)}", file=sys.stderr)
            return 1
    failed: list[str] = []
    for i, source in enumerate(sources, start=1):
        print(f"[{i}/{len(sources)}] {source.title} ... ", end="", flush=True)
        try:
            path = scraper.scrape_source(source, SETTINGS.raw_dir)
            print(f"ок → {path.name}")
        except Exception as exc:  # noqa: BLE001 — сообщаем и продолжаем корпус
            print(f"ОШИБКА: {exc}")
            failed.append(source.key)
    if failed:
        print(f"\nНе удалось загрузить: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _prune_removed_sources(conn, sources: list) -> list[str]:
    """Удаляет из базы НПА, чьего adilet_url больше нет в sources.yaml —
    например, после исправления неверного URL (см. Finding 2). Без этого
    старый акт (в т.ч. утративший силу) остаётся в базе бессрочно рядом с
    новым. Каскад (ON DELETE CASCADE + триггеры) сам чистит chunks,
    vec_chunks и chunks_fts."""
    valid_urls = {s.url for s in sources}
    removed = []
    for act_id, title, url in conn.execute("SELECT id, title, adilet_url FROM acts").fetchall():
        if url not in valid_urls:
            conn.execute("DELETE FROM acts WHERE id = ?", (act_id,))
            removed.append(title)
    if removed:
        conn.commit()
    return removed


def cmd_index() -> int:
    llm.health_check()
    conn = db.connect()
    db.create_schema(conn)
    if not SETTINGS.raw_dir.exists() or not any(SETTINGS.raw_dir.glob("*.json")):
        print(
            f"В {SETTINGS.raw_dir} нет данных. Сначала выполните: python cli.py scrape",
            file=sys.stderr,
        )
        return 1
    for key, count in indexer.index_directory(conn, SETTINGS.raw_dir).items():
        print(f"{key}: {count} чанков")
    removed = _prune_removed_sources(conn, load_sources())
    if removed:
        print(f"Удалены НПА, отсутствующие в sources.yaml: {', '.join(removed)}")
    return 0


def cmd_status() -> int:
    conn = db.connect()
    db.create_schema(conn)
    acts = db.corpus_summary(conn)
    if not acts:
        print("Индекс пуст. Выполните: python cli.py scrape && python cli.py index")
        return 1
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"НПА в индексе: {len(acts)}, чанков: {total}\n")
    for title, revision in acts:
        print(f"  {title} — ред. {revision or 'не определена'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Юридический консультант по НПА РК")
    sub = parser.add_subparsers(dest="command", required=True)
    scrape = sub.add_parser("scrape", help="загрузить НПА с adilet.zan.kz")
    scrape.add_argument("--only", nargs="*", help="ключи источников из sources.yaml")
    sub.add_parser("index", help="собрать индекс из загруженных данных")
    sub.add_parser("status", help="показать состояние индекса")
    args = parser.parse_args()

    if args.command == "scrape":
        return cmd_scrape(args.only)
    if args.command == "index":
        return cmd_index()
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main())

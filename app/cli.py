import argparse
import asyncio
import sys
from collections.abc import Sequence

from sqlalchemy.exc import OperationalError

from app.auth.bootstrap import bootstrap_admin
from app.auth.service import TenantInactiveError
from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory


async def _create_admin(database_url: str, tenant: str, name: str) -> tuple[str, str, str]:
    engine = create_engine(database_url)
    try:
        async with create_session_factory(engine)() as session:
            found, key, secret = await bootstrap_admin(session, tenant, name)
            return found.name, key.key_prefix, secret
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None, database_url: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="LLMRouter admin tools")
    commands = parser.add_subparsers(dest="command", required=True)
    admin = commands.add_parser("create-admin", help="mint an admin API key (shown once)")
    admin.add_argument("--tenant", default="platform", help="tenant that owns the key")
    admin.add_argument("--name", default="bootstrap admin", help="label for the key")
    args = parser.parse_args(argv)

    url = database_url or get_settings().database_url
    try:
        tenant, prefix, secret = asyncio.run(_create_admin(url, args.tenant, args.name))
    except OperationalError as exc:
        print(
            f"error: the database is not ready; run `make migrate` first ({exc.orig})",
            file=sys.stderr,
        )
        return 1
    except TenantInactiveError:
        print(f"error: tenant '{args.tenant}' is deactivated", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Created admin key {prefix}... for tenant '{tenant}'.")
    print("Copy it now; it cannot be shown again:\n")
    print(secret)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

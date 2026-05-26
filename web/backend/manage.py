#!/usr/bin/env python3
"""Small admin CLI for the web layer.

Examples:
    python -m web.backend.manage create-user alice s3cret
    python -m web.backend.manage create-user bob s3cret --admin
    python -m web.backend.manage reset-password admin newpass
    python -m web.backend.manage list-users
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python web/backend/manage.py ...` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.db import SessionLocal, init_db  # noqa: E402
from app.models import User  # noqa: E402
from app.security import hash_password  # noqa: E402


def cmd_create_user(args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == args.username).first():
            print(f"User '{args.username}' already exists", file=sys.stderr)
            return 2
        db.add(User(
            username=args.username,
            password_hash=hash_password(args.password),
            is_admin=1 if args.admin else 0,
        ))
        db.commit()
        print(f"Created {'admin' if args.admin else 'user'} '{args.username}'")
        return 0
    finally:
        db.close()


def cmd_reset_password(args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == args.username).one_or_none()
        if not user:
            print(f"User '{args.username}' not found", file=sys.stderr)
            return 2
        user.password_hash = hash_password(args.password)
        db.commit()
        print(f"Reset password for '{args.username}'")
        return 0
    finally:
        db.close()


def cmd_list_users(_: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        for u in db.query(User).order_by(User.id).all():
            tag = " [admin]" if u.is_admin else ""
            print(f"  {u.id:>3}  {u.username}{tag}  created={u.created_at:%Y-%m-%d}")
        return 0
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser(description="DeckWeaver web admin CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    cu = sub.add_parser("create-user")
    cu.add_argument("username")
    cu.add_argument("password")
    cu.add_argument("--admin", action="store_true")
    cu.set_defaults(func=cmd_create_user)

    rp = sub.add_parser("reset-password")
    rp.add_argument("username")
    rp.add_argument("password")
    rp.set_defaults(func=cmd_reset_password)

    lu = sub.add_parser("list-users")
    lu.set_defaults(func=cmd_list_users)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

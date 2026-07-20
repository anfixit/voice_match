"""Тесты CLI-точки входа."""

from voice_match import __main__ as entrypoint
from voice_match.ui import interface


def test_check_environment_reports_missing_package(monkeypatch) -> None:
    monkeypatch.setattr(
        entrypoint.importlib.util,
        'find_spec',
        lambda name: None if name == 'torch' else object(),
    )

    assert entrypoint.check_environment() == ('torch',)


def test_main_launches_ui(monkeypatch) -> None:
    launched = False

    def fake_launch() -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr(entrypoint, 'check_environment', lambda: ())
    monkeypatch.setattr(
        type(entrypoint.settings),
        'ensure_dirs',
        lambda self: None,
    )
    monkeypatch.setattr(interface, 'launch_ui', fake_launch)

    entrypoint.main()

    assert launched

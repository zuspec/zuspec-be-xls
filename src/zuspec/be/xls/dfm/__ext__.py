"""Registers the zuspec-be-xls dv-flow package."""
import os


def dvfm_packages():
    return {"zuspec.xls": os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.dv")}

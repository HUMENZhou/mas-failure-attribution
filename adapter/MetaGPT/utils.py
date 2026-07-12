"""Name/profile mapping helpers for MetaGPT role normalization."""

NAME_PROFILE_MAP = {
    'Mike': 'Team Leader',
    'Alex': 'Engineer',
    'David': 'DataAnalyst',
    'Bob': 'Architect',
    'Alice': 'Product Manager',
    'metagpt.roles.searcher.Searcher':'Searcher',
}

PROFILE_NAME_MAP = {
    'Team Leader': 'Mike',
    'Engineer': 'Alex',
    'DataAnalyst': 'David',
    'Architect': 'Bob',
    'Product Manager': 'Alice',
    'metagpt.roles.searcher.Searcher':'Sam',
}

def get_profile(name: str):
    """Convert MetaGPT internal role name to canonical profile label."""
    return NAME_PROFILE_MAP[name]

def get_name(profile: str):
    """Convert canonical profile label to MetaGPT internal role name."""
    return PROFILE_NAME_MAP[profile]
"""Template-based dataset augmentation.

Combines the hand-written seed examples from `build_dataset.py` (35 per
category) with a much larger pool of template x filler combinations per
category, deduplicates everything (including against
`data/regression_dataset.jsonl`, so the held-out regression set never leaks
into training), and writes the combined result to
`data/training_data.jsonl`.

Not part of the public API or the runtime training pipeline itself -- a
one-off, reproducible (fixed random seed) content generator kept so the
larger dataset is auditable and regenerable. Safe to re-run.

Usage:
    python training/augment_dataset.py [--new-per-category 305] [--seed 42]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from build_dataset import LABELED as SEED_LABELED

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Shared filler vocabularies
# ---------------------------------------------------------------------------

COUNTRIES = [
    "France", "Germany", "Italy", "Spain", "Portugal", "Japan", "China",
    "India", "Brazil", "Canada", "Australia", "Mexico", "Russia", "Egypt",
    "Kenya", "Nigeria", "Argentina", "Chile", "Peru", "Colombia", "Sweden",
    "Norway", "Finland", "Denmark", "the Netherlands", "Belgium",
    "Switzerland", "Austria", "Greece", "Turkey", "Poland", "Ukraine",
    "South Korea", "Thailand", "Vietnam", "Indonesia", "the Philippines",
    "New Zealand", "Ireland", "South Africa", "Morocco", "Iceland",
    "Hungary", "Czechia",
]

OPPOSITE_WORDS = [
    "hot", "cold", "fast", "slow", "big", "small", "happy", "sad", "light",
    "dark", "strong", "weak", "easy", "difficult", "full", "empty", "old",
    "young", "rich", "poor", "clean", "dirty", "open", "closed", "loud",
    "quiet", "early", "late", "wet", "dry", "high", "low", "thick", "thin",
    "wide", "narrow", "near", "far", "soft", "sharp", "brave", "shy",
]

PLURAL_WORDS = [
    "mouse", "goose", "child", "foot", "tooth", "cactus", "fungus", "index",
    "matrix", "vertex", "ox", "person", "leaf", "knife", "wife", "half",
    "potato", "tomato", "hero", "echo", "sheep", "fish", "deer", "species",
    "series", "analysis", "crisis", "phenomenon", "criterion", "datum",
]

SHAPES = [
    "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon",
    "nonagon", "decagon", "rectangle", "rhombus", "trapezoid",
    "parallelogram",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SPELL_WORDS = [
    "necessary", "separate", "definitely", "occurrence", "embarrass",
    "accommodate", "rhythm", "conscience", "questionnaire", "maintenance",
    "liaison", "broccoli", "bureaucracy", "entrepreneur", "millennium",
]

BOOKS = [
    "Romeo and Juliet", "Pride and Prejudice", "Moby Dick", "War and Peace",
    "1984", "The Great Gatsby", "Hamlet", "The Odyssey", "Don Quixote",
    "Crime and Punishment", "Frankenstein", "Dracula",
    "The Catcher in the Rye", "To Kill a Mockingbird", "Jane Eyre",
]

PAINTINGS = [
    "the Mona Lisa", "Starry Night", "The Scream", "Guernica",
    "The Persistence of Memory", "Girl with a Pearl Earring",
    "The Last Supper", "American Gothic",
]

EVENTS_YEAR = [
    "World War II end", "World War I start", "the Berlin Wall fall",
    "the moon landing happen", "the Titanic sink",
    "the French Revolution start", "the internet become publicly available",
]

DISTANCE_UNITS = [
    ("kilometers", "miles"), ("miles", "kilometers"), ("feet", "inches"),
    ("inches", "centimeters"), ("meters", "feet"), ("yards", "meters"),
]
WEIGHT_UNITS = [("pounds", "kilograms"), ("ounces", "grams"), ("kilograms", "pounds")]
VOLUME_UNITS = [("gallons", "liters"), ("liters", "gallons")]
TEMP_UNITS = [("Celsius", "Fahrenheit"), ("Fahrenheit", "Celsius")]
CONVERSION_UNITS = DISTANCE_UNITS + WEIGHT_UNITS + VOLUME_UNITS + TEMP_UNITS

TIME_UNIT_PAIRS = [
    ("minutes", "hours"), ("hours", "days"), ("days", "weeks"),
    ("weeks", "months"), ("seconds", "minutes"),
]


def _n(rng: random.Random, lo: int = 2, hi: int = 99) -> int:
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# SIMPLE
# ---------------------------------------------------------------------------

def generate_simple(rng: random.Random) -> list[str]:
    out: list[str] = []

    for country in COUNTRIES:
        out.append(f"What is the capital of {country}?")
        out.append(f"What is the currency used in {country}?")
        out.append(f"What language is spoken in {country}?")

    for word in OPPOSITE_WORDS:
        out.append(f"What is the opposite of {word}?")
    for word in PLURAL_WORDS:
        out.append(f"What is the plural of {word}?")
    for shape in SHAPES:
        out.append(f"How many sides does a {shape} have?")
    for day in DAYS:
        out.append(f"What day comes after {day}?")
        out.append(f"What day comes before {day}?")
    for word in SPELL_WORDS:
        out.append(f"Spell the word {word}.")
    for book in BOOKS:
        out.append(f"Who wrote {book}?")
    for painting in PAINTINGS:
        out.append(f"Who painted {painting}?")
    for event in EVENTS_YEAR:
        out.append(f"What year did {event}?")
    for scale in ("Celsius", "Fahrenheit"):
        out.append(f"What is the boiling point of water in {scale}?")
        out.append(f"What is the freezing point of water in {scale}?")

    for _ in range(400):
        a, b = _n(rng), _n(rng)
        n_from, n_to = rng.choice(CONVERSION_UNITS)
        out.append(f"Convert {a} {n_from} to {n_to}.")

    for _ in range(400):
        a, b = _n(rng), _n(rng)
        op = rng.choice(["plus", "minus", "times", "divided by"])
        out.append(f"What is {a} {op} {b}?")

    for _ in range(150):
        pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
        base = _n(rng, 10, 999)
        out.append(f"What is {pct} percent of {base}?")

    for sq in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289]:
        out.append(f"What is the square root of {sq}?")

    for _ in range(150):
        small, big = rng.choice(TIME_UNIT_PAIRS)
        n = rng.randint(2, 12)
        out.append(f"How many {small} are in {n} {big}?")

    return out


# ---------------------------------------------------------------------------
# CODING
# ---------------------------------------------------------------------------

LANGS = [
    "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "TypeScript",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala",
]

CODE_TASKS = [
    "reverse a string", "sort a list of integers", "merge two dictionaries",
    "find duplicate values in an array", "parse a CSV file",
    "flatten a nested list", "check if a string is a palindrome",
    "count word frequency in a text", "remove whitespace from a string",
    "validate an email address", "generate a random password",
    "calculate the factorial of a number",
    "find the longest common substring between two strings",
    "implement a stack using an array",
    "implement a queue using two stacks",
    "detect a cycle in a directed graph",
    "compute the nth Fibonacci number", "binary search a sorted array",
    "deep clone a nested object", "debounce a function call",
    "rotate an array by k positions", "find the missing number in a sequence",
    "convert a string to title case", "implement a rate limiter",
    "serialize an object to JSON", "deserialize JSON into an object",
    "find the intersection of two arrays", "implement an LRU cache",
    "shuffle a list randomly", "group a list of objects by a key",
    "chunk an array into fixed-size batches",
]

DEBUG_ISSUES = [
    "throws a null pointer exception", "causes a stack overflow",
    "runs into an infinite loop", "returns the wrong output for negative numbers",
    "crashes on empty input", "leaks memory over time",
    "fails silently without an error",
    "produces inconsistent results across runs",
    "hangs when given large input", "throws an index out of range error",
]

CONCEPTS = [
    "closures", "decorators", "garbage collection", "async/await",
    "generators", "dependency injection", "memoization", "promises",
    "context managers", "multithreading", "the event loop",
    "type inference", "reflection", "operator overloading",
    "lazy evaluation",
]

FRAMEWORKS = [
    "React", "Django", "Flask", "Express", "Spring", "Angular", "Vue",
    "FastAPI", "Rails", "Next.js",
]
FRAMEWORK_COMPONENTS = [
    "component", "hook", "middleware", "serializer", "controller",
    "service", "reducer", "route handler", "form", "modal",
]
FRAMEWORK_PROBLEMS = [
    "re-rendering infinitely", "not updating on state change",
    "throwing a hydration error", "failing to submit",
    "not receiving props", "crashing on mount",
    "leaking event listeners", "not unmounting cleanly",
]

SQL_TASKS = [
    "find duplicate rows in a table", "count orders per customer",
    "calculate the running total of sales",
    "find the second highest salary", "join three tables on a shared key",
    "find customers with no orders",
    "calculate month-over-month growth",
    "find the top 5 products by revenue",
    "detect gaps in a sequence of IDs", "pivot rows into columns",
]

HOW_DO_I_ACTIONS = [
    "connect to a PostgreSQL database", "set up unit testing",
    "handle exceptions in a try-except block",
    "read a large file line by line", "parse command-line arguments",
    "schedule a background task", "implement pagination for an API",
    "cache the result of an expensive function", "set up structured logging",
    "mock an external API call in tests",
]

COMPARE_PAIRS_CODE = [
    "a list and a tuple", "a stack and a queue",
    "an interface and an abstract class", "REST and GraphQL",
    "SQL and NoSQL databases", "a process and a thread",
    "synchronous and asynchronous code", "a shallow copy and a deep copy",
    "composition and inheritance", "a compiler and an interpreter",
]

ALGORITHMS = [
    "quicksort", "mergesort", "binary search", "depth-first search",
    "breadth-first search", "dynamic programming for the knapsack problem",
    "Dijkstra's algorithm", "A* search", "topological sort",
    "the sliding window technique", "the two-pointer technique",
    "heap sort", "radix sort", "union-find", "the Boyer-Moore algorithm",
]

ERROR_TYPES = [
    "segmentation fault", "null pointer exception",
    "index out of bounds error", "type error", "memory leak",
    "race condition", "deadlock", "stack overflow", "off-by-one error",
    "null reference exception",
]

APP_TYPES = [
    "Flask", "Django", "FastAPI", "Express", "Spring Boot", "Node.js",
    "React", "static website", "microservice", "background worker",
]

PATTERN_TARGETS = [
    "an email address", "a phone number", "a US zip code", "a URL",
    "a credit card number", "an IPv4 address", "a strong password",
    "a hex color code", "a date in YYYY-MM-DD format", "a UUID",
]

SLOW_QUERIES = [
    "SQL join", "SQL query with multiple subqueries",
    "MongoDB aggregation pipeline", "GraphQL resolver",
    "Elasticsearch query",
]


def generate_coding(rng: random.Random) -> list[str]:
    out: list[str] = []

    for lang, task in itertools.product(LANGS, CODE_TASKS):
        out.append(f"Write a {lang} function to {task}.")

    for lang, issue in itertools.product(LANGS, DEBUG_ISSUES):
        out.append(f"Debug this {lang} code that {issue}.")

    for lang, concept in itertools.product(LANGS, CONCEPTS):
        out.append(f"Explain how {concept} works in {lang}.")

    for framework in FRAMEWORKS:
        for component in FRAMEWORK_COMPONENTS:
            problem = rng.choice(FRAMEWORK_PROBLEMS)
            out.append(f"Why is my {framework} {component} {problem}?")

    for task in SQL_TASKS:
        out.append(f"Write a SQL query to {task}.")

    for lang, action in itertools.product(LANGS, HOW_DO_I_ACTIONS):
        out.append(f"How do I {action} in {lang}?")

    for pair, lang in itertools.product(COMPARE_PAIRS_CODE, LANGS):
        out.append(f"What is the difference between {pair} in {lang}?")

    for algorithm, lang in itertools.product(ALGORITHMS, LANGS):
        out.append(f"Implement {algorithm} in {lang}.")
    for algorithm in ALGORITHMS:
        out.append(f"Explain the time complexity of {algorithm}.")

    for error_type, lang in itertools.product(ERROR_TYPES, LANGS):
        out.append(f"Fix this {error_type} in my {lang} code.")

    for app_type in APP_TYPES:
        out.append(f"Write a Dockerfile for a {app_type} application.")

    for pattern in PATTERN_TARGETS:
        out.append(f"Write a regex to validate {pattern}.")

    for q in SLOW_QUERIES:
        out.append(f"How do I optimize this slow {q}?")

    for _ in range(100):
        lang = rng.choice(LANGS)
        task = rng.choice(CODE_TASKS)
        out.append(f"Refactor this {lang} function that {rng.choice(DEBUG_ISSUES)}.")
        out.append(f"Write unit tests for the {lang} function that {task}.")

    return out


# ---------------------------------------------------------------------------
# REASONING
# ---------------------------------------------------------------------------

TRADEOFF_PAIRS = [
    ("microservices", "a monolithic architecture"),
    ("remote work", "in-office work"),
    ("renting", "buying a home"),
    ("nuclear energy", "fossil fuels"),
    ("a greedy algorithm", "dynamic programming"),
    ("strict consistency", "eventual consistency"),
    ("horizontal scaling", "vertical scaling"),
    ("a relational database", "a NoSQL database"),
    ("outsourcing", "building in-house"),
    ("organic growth", "growth through acquisition"),
    ("debt financing", "equity financing"),
    ("a fixed exchange rate", "a floating exchange rate"),
    ("centralized planning", "free markets"),
    ("public transit investment", "road expansion"),
    ("early retirement", "continuing to work"),
    ("a subscription model", "a one-time purchase model"),
    ("open-plan offices", "private offices"),
    ("standardized testing", "holistic admissions"),
    ("universal basic income", "targeted welfare programs"),
    ("nuclear power", "renewable energy"),
]

REASONING_DOMAINS = [
    "distributed systems", "personal finance", "urban planning",
    "supply chain management", "product strategy", "software architecture",
    "public policy", "team management", "clinical trial design",
    "portfolio management",
]

CAUSAL_CLAIMS = [
    "correlation does not imply causation",
    "diversifying a portfolio reduces risk",
    "a larger sample size reduces statistical uncertainty",
    "sunk costs shouldn't influence future decisions",
    "compound interest causes exponential growth",
    "minimum wage increases can reduce entry-level jobs",
    "specialization increases economic efficiency",
    "redundancy improves system reliability",
    "premature optimization is often wasteful",
    "information asymmetry can lead to market failure",
    "network effects create winner-take-all markets",
    "survivorship bias skews success stories",
    "regression to the mean affects extreme outcomes",
    "Goodhart's law undermines metrics used as targets",
    "the tragedy of the commons depletes shared resources",
]

CAUSAL_EVENTS = [
    "the fall of the Roman Empire", "the 2008 financial crisis",
    "the dot-com bubble burst", "the fall of the Berlin Wall",
    "the collapse of a major bridge",
    "a cascading failure in a distributed system",
    "a well-tested system failing in production",
    "a company's stock price crashing overnight",
    "a supply chain shortage",
    "a viral product's sudden decline in popularity",
]

POLICIES = [
    "raising interest rates", "increasing the minimum wage",
    "imposing tariffs on imported goods", "cutting corporate tax rates",
    "introducing a carbon tax", "expanding unemployment benefits",
    "deregulating an industry", "enforcing stricter data privacy laws",
    "subsidizing renewable energy", "raising the retirement age",
]

ENTITY_GOALS = [
    ("a startup", "growth", "profitability"),
    ("a nonprofit", "impact", "fundraising"),
    ("a public company", "short-term earnings", "long-term investment"),
    ("a research lab", "publishing quickly", "rigorous validation"),
    ("a hospital", "cost control", "patient outcomes"),
    ("a school district", "standardized test scores", "holistic student development"),
]

TRENDS = [
    "automation", "remote work", "AI adoption", "globalization",
    "climate change", "an aging population", "rising interest rates",
    "increased regulation", "rapid urbanization", "the gig economy",
]

MECHANISM_OUTCOME = [
    ("compound interest", "exponential growth in savings"),
    ("network effects", "winner-take-all market dynamics"),
    ("positive feedback loops", "runaway climate change"),
    ("herd immunity", "reduced disease transmission"),
    ("economies of scale", "falling per-unit costs"),
    ("diminishing returns", "slowing output growth"),
    ("information cascades", "market bubbles"),
    ("moral hazard", "excessive risk-taking"),
]


def generate_reasoning(rng: random.Random) -> list[str]:
    out: list[str] = []

    for a, b in TRADEOFF_PAIRS:
        out.append(f"Compare the trade-offs between {a} and {b}.")
        domain = rng.choice(REASONING_DOMAINS)
        out.append(f"Weigh the trade-offs between {a} and {b} in {domain}.")

    for claim in CAUSAL_CLAIMS:
        out.append(f"Explain the reasoning behind why {claim}.")

    for event in CAUSAL_EVENTS:
        out.append(f"Analyze the causes that led to {event}.")

    for policy in POLICIES:
        out.append(f"What are the second-order effects of {policy}?")

    for entity, goal1, goal2 in ENTITY_GOALS:
        out.append(f"Evaluate whether {entity} should prioritize {goal1} or {goal2} first.")

    for trend, domain in itertools.product(TRENDS, REASONING_DOMAINS):
        out.append(f"Discuss the implications of {trend} on {domain}.")

    for mechanism, outcome in MECHANISM_OUTCOME:
        out.append(f"Explain step by step how {mechanism} causes {outcome}.")

    for _ in range(120):
        s1, s2 = rng.randint(30, 90), rng.randint(30, 90)
        dist = rng.randint(100, 600)
        out.append(
            f"A train leaves station A at {s1} mph and another leaves station B "
            f"at {s2} mph heading toward each other on a {dist}-mile track; "
            f"when do they meet?"
        )

    for _ in range(120):
        n1 = rng.randint(2, 20)
        n3 = rng.randint(n1 + 1, n1 + 50)
        out.append(
            f"If it takes {n1} machines {n1} minutes to make {n1} widgets, "
            f"how long would {n3} machines take to make {n3} widgets?"
        )

    for _ in range(80):
        h1, h2 = rng.randint(2, 10), rng.randint(2, 10)
        out.append(
            f"If two pipes fill a tank in {h1} and {h2} hours respectively, "
            f"how long would it take together?"
        )

    for _ in range(80):
        n = rng.randint(4, 9)
        avg = rng.randint(10, 90)
        out.append(
            f"The average of {n} numbers is {avg}; if {n - 1} of them are "
            f"known, how do you find the last one?"
        )

    for _ in range(80):
        principal = rng.choice([1000, 2000, 5000, 10000, 25000])
        rate = rng.choice([2, 3, 4, 5, 6, 7, 8])
        years = rng.randint(2, 20)
        out.append(
            f"An investment of ${principal} grows at {rate} percent annual "
            f"compound interest; what is it worth after {years} years?"
        )

    return out


# ---------------------------------------------------------------------------
# SECURITY
# ---------------------------------------------------------------------------

SEC_SYSTEMS = [
    "login authentication", "password reset", "session management",
    "file upload", "payment processing", "user registration",
    "API key management", "OAuth2 authorization", "single sign-on",
    "password storage",
]
SEC_COMPONENTS = [
    "endpoint", "handler", "flow", "module", "middleware", "service",
    "form", "API", "function", "configuration",
]

SEC_TECHS = [
    "Express.js", "Django", "Flask", "Spring Boot", "GraphQL",
    "REST", "Node.js", "PHP", "Ruby on Rails", "a serverless Lambda",
]
VULN_TYPES = [
    "SQL injection", "cross-site scripting", "command injection",
    "insecure deserialization", "server-side request forgery",
    "XML external entity", "prototype pollution", "path traversal",
    "broken access control", "cross-site request forgery",
]

SEC_TARGETS = [
    "input field", "API endpoint", "file parser", "login form",
    "comment section", "search bar", "upload handler", "URL parameter",
    "HTTP header", "cookie",
]

ARTIFACTS = [
    "Dockerfile", "Terraform configuration", "Kubernetes manifest",
    "Nginx configuration", "CI/CD pipeline definition", "AWS IAM policy",
    "OAuth2 flow", "smart contract", "password reset flow",
    "session management implementation",
]
ISSUE_TYPES = [
    "insecure defaults", "overly permissive permissions",
    "hardcoded secrets", "missing input validation", "weak cryptography",
    "improper error handling", "insufficient logging",
    "race conditions", "insecure deserialization", "broken authentication",
]

SECURING_ACTIONS = [
    "storing user passwords", "managing API keys",
    "handling file uploads", "implementing session tokens",
    "encrypting data at rest", "validating user input",
    "configuring CORS", "rotating secrets",
    "logging sensitive operations", "authenticating microservices",
]

CONFIG_TYPES = [
    "AWS S3 bucket policy", "Kubernetes RBAC configuration",
    "Nginx server configuration", "Terraform IAM role",
    "firewall rule set", "CORS configuration", "TLS configuration",
    "VPC network configuration", "database access control list",
    "API gateway configuration",
]
RISKS = [
    "public read access", "privilege escalation",
    "overly permissive origins", "excessive permissions", "open ports",
    "information disclosure", "unauthenticated access",
    "weak cipher suites", "cross-tenant data leakage",
    "insecure default credentials",
]

MITIGATIONS = [
    "rate limiting", "input sanitization", "parameterized queries",
    "content security policy headers", "multi-factor authentication",
    "least-privilege access control", "TLS certificate pinning",
    "secure session cookies", "dependency vulnerability scanning",
    "a web application firewall",
]
ATTACK_TYPES = [
    "a denial-of-service attack", "a brute-force login attack",
    "a credential stuffing attack", "an SQL injection attack",
    "a cross-site scripting attack", "a man-in-the-middle attack",
    "a session hijacking attack", "a privilege escalation attack",
    "a supply chain attack", "a phishing-based account takeover",
]


def generate_security(rng: random.Random) -> list[str]:
    out: list[str] = []

    for system, component in itertools.product(SEC_SYSTEMS, SEC_COMPONENTS):
        out.append(f"Find security weaknesses in this {system} {component}.")

    for tech, vuln in itertools.product(SEC_TECHS, VULN_TYPES):
        component = rng.choice(SEC_COMPONENTS)
        out.append(f"Is this {tech} {component} vulnerable to {vuln}?")

    for vuln, target in itertools.product(VULN_TYPES, SEC_TARGETS):
        out.append(f"Explain how a {vuln} attack could exploit this {target}.")

    for artifact, issue in itertools.product(ARTIFACTS, ISSUE_TYPES):
        out.append(f"Review this {artifact} for {issue}.")

    for action in SECURING_ACTIONS:
        out.append(f"What are best practices for securely {action}?")

    for config, risk in itertools.product(CONFIG_TYPES, RISKS):
        out.append(f"Audit this {config} for {risk}.")

    for mitigation, attack in itertools.product(MITIGATIONS, ATTACK_TYPES):
        out.append(f"Explain how {mitigation} would help prevent {attack} against this system.")

    return out


# ---------------------------------------------------------------------------
# SUMMARIZATION
# ---------------------------------------------------------------------------

DOC_TYPES = [
    "article", "research paper", "legal contract", "meeting notes",
    "quarterly earnings report", "email thread", "technical whitepaper",
    "changelog", "court ruling", "support ticket thread",
    "interview transcript", "policy document",
    "GitHub pull request description", "webinar transcript",
    "financial statement", "opinion essay", "press release",
    "product documentation", "customer feedback survey",
    "incident postmortem", "podcast transcript", "book chapter",
    "news article", "business proposal", "vendor agreement",
    "user manual", "release notes", "conference talk transcript",
    "internal memo", "design document",
]

SUMMARY_LENGTHS = [
    "three sentences", "one paragraph", "a single sentence",
    "two sentences", "a short bullet list", "under 100 words",
    "a brief executive summary",
]

SUMMARY_OUTPUTS = [
    "key action items", "the main discussion points", "the core issue",
    "the central argument", "three key insights",
    "a one-line issue description", "the key clauses",
    "the main arguments", "the key findings", "the decisions made",
]


def generate_summarization(rng: random.Random) -> list[str]:
    out: list[str] = []

    for doc in DOC_TYPES:
        length = rng.choice(SUMMARY_LENGTHS)
        output = rng.choice(SUMMARY_OUTPUTS)
        out.append(f"Summarize this {doc} in {length}.")
        out.append(f"Give me a TL;DR of this {doc}.")
        out.append(f"Condense this {doc} into {output}.")
        out.append(f"Provide a {length} summary of this {doc}.")
        out.append(f"Extract the key takeaways from this {doc}.")
        out.append(f"Give a concise overview of this {doc}.")
        out.append(f"Summarize the main points of this {doc}.")

    for doc, length in itertools.product(DOC_TYPES, SUMMARY_LENGTHS):
        out.append(f"Provide a {length} recap of this {doc}.")

    return out


GENERATORS = {
    "simple": generate_simple,
    "coding": generate_coding,
    "reasoning": generate_reasoning,
    "security": generate_security,
    "summarization": generate_summarization,
}


def _normalize(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def build_augmented_dataset(new_per_category: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)

    excluded = {_normalize(t) for texts in SEED_LABELED.values() for t in texts}
    regression_path = DATA_DIR / "regression_dataset.jsonl"
    if regression_path.is_file():
        with regression_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    excluded.add(_normalize(json.loads(line)["text"]))

    result: dict[str, list[str]] = {}
    for label, generator in GENERATORS.items():
        candidates = generator(rng)

        seen_in_pool: set[str] = set()
        unique_candidates: list[str] = []
        for text in candidates:
            key = _normalize(text)
            if key in excluded or key in seen_in_pool:
                continue
            seen_in_pool.add(key)
            unique_candidates.append(text)

        rng.shuffle(unique_candidates)
        if len(unique_candidates) < new_per_category:
            raise RuntimeError(
                f"category '{label}': only generated {len(unique_candidates)} unique "
                f"candidates, need {new_per_category}. Add more templates/fillers."
            )
        result[label] = SEED_LABELED[label] + unique_candidates[:new_per_category]

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment the llm_router training dataset.")
    parser.add_argument("--new-per-category", type=int, default=305)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "training_data.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = build_augmented_dataset(args.new_per_category, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for label, texts in combined.items():
            for text in texts:
                fh.write(json.dumps({"text": text, "label": label}) + "\n")

    total = sum(len(v) for v in combined.values())
    new_total = sum(len(v) - len(SEED_LABELED[k]) for k, v in combined.items())
    print(f"wrote {total} examples ({new_total} newly generated) to {args.output}")
    for label, texts in combined.items():
        print(f"  {label}: {len(texts)} total")


if __name__ == "__main__":
    main()

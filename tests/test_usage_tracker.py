from app.usage.tracker import InMemoryUsageTracker


def test_increment_starts_at_one():
    tracker = InMemoryUsageTracker()
    assert tracker.increment("cust-1") == 1


def test_increment_accumulates_per_customer():
    tracker = InMemoryUsageTracker()
    tracker.increment("cust-1")
    tracker.increment("cust-1")
    tracker.increment("cust-1")
    assert tracker.get_count("cust-1") == 3


def test_customers_are_isolated():
    tracker = InMemoryUsageTracker()
    tracker.increment("cust-1")
    tracker.increment("cust-2")
    tracker.increment("cust-2")
    assert tracker.get_count("cust-1") == 1
    assert tracker.get_count("cust-2") == 2


def test_unknown_customer_has_zero_count():
    tracker = InMemoryUsageTracker()
    assert tracker.get_count("never-called") == 0

import time
from rich.live import Live
from rich.table import Table

def generate_test_table(metric_dict):
    positive = metric_dict.get('test_success', 0)
    negative = metric_dict.get('test_failure', 0)
    table = Table(title="Live Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    if positive==0 and negative==0: return table
    table.add_row("Success times:", f"{positive}")
    table.add_row("Failer times:", f"{negative}")
    sr=positive/(positive+negative)
    table.add_row("Success rate %:", f"{sr}%")
    table.add_row("Number of scenes", f"{positive+negative}")
    return table

if __name__ == '__main__':
    with Live(generate_test_table(), refresh_per_second=4, screen=True) as live:
        while True:
            time.sleep(0.01)
            live.update(generate_test_table(0,0))
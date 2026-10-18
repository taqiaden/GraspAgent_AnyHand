import time
from rich.live import Live
from rich.table import Table
from rich.console import Group
from rich.text import Text
from rich.progress_bar import ProgressBar

def generate_combined_table(metric_dict, step, iterations,hand_name=''):
    return Group(
        generate_test_table(metric_dict, hand_name=hand_name),
        generate_training_table(metric_dict, hand_name=hand_name),
        generate_progress_bar(step, iterations),
    )

def generate_progress_bar(step, iterations):
    total = max(iterations, 1)
    current = min(step, total)

    header = Text(
        f"Test progress: {current}/{total}  ", justify="left"
    )
    bar = ProgressBar(total=total, completed=current,
                      width=51)
    return Group(header, bar)

def generate_test_table(metric_dict,hand_name=''):
    positive = metric_dict.get('test_success', 0)
    negative = metric_dict.get('test_failure', 0)
    table = Table(title="Live Test Metrics for "+hand_name)
    table.add_column("Metric",width=30)
    table.add_column("Value",width=15, justify="center")
    if positive==0 and negative==0: return table
    table.add_row("Success times:", f"{positive}")
    table.add_row("Failer times:", f"{negative}")
    sr=positive/(positive+negative)
    table.add_row("Success rate %:", f"{sr}%")
    table.add_row("Total number of trials", f"{positive+negative}")
    return table

def generate_training_table(metric_dict,hand_name=''):

    table = Table(title="Training Last Metrics for "+hand_name,)
    table.add_column("Metric",width=30)
    table.add_column("Value",width=15, justify="center")
    table.add_row("Coverage Indicator:", f"{metric_dict['uniqueness']}")
    table.add_row("Flexibility Indicator:", f"{metric_dict['flexibility']}")
    table.add_row("Discrimination Indicator:", f"{metric_dict['discrimination_dist']}")
    table.add_row("Confidence Indictor:", f"{metric_dict['confidence']}")
    table.add_row("Heuristic Efficiency:", f"{metric_dict['heuristic_efficiency']}")
    table.add_row("Buffer Replay Rate:", f"{metric_dict['replay_rate']}")

    return table

if __name__ == '__main__':
    with Live(generate_test_table(), refresh_per_second=4, screen=True) as live:
        while True:
            time.sleep(0.01)
            live.update(generate_test_table(0,0))
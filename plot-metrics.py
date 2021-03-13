import json
import sys

import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open(sys.argv[1], 'rt') as fd:
        with open(sys.argv[2], 'rt') as fd2:
            metrics_full = json.load(fd)[1:-1]
            metrics_120 = json.load(fd2)[:-1]
            assert(len(metrics_120) == len(metrics_full))
            x = list(range(1, len(metrics_full) + 1))
            full = (
                x,
                [m['eval_precision'] for m in metrics_full],
                x,
                [m['eval_recall'] for m in metrics_full],
                x,
                [m['eval_f1'] for m in metrics_full],
            )
            part = (
                x,
                [m['eval_precision'] for m in metrics_120],
                x,
                [m['eval_recall'] for m in metrics_120],
                x,
                [m['eval_f1'] for m in metrics_120],
            )

            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
            fig3, ax3 = plt.subplots()
            ax1.set_title('Performance on full dataset')
            ax2.set_title('Performance on 120 samples')
            ax3.set_title('Performance comparison')
            ax1.set_xticks(range(1, 13))
            ax2.set_xticks(range(1, 13))
            ax3.set_xticks(range(1, 13))
            ax1.set_ylim(0.85, 1.0)
            ax2.set_ylim(0.85, 1.0)
            ax3.set_ylim(0.85, 1.0)
            ax1.plot(*full)
            ax2.plot(*part)
            ax3.plot(x, full[5], x, part[5])
            ax1.set_xlabel("Layers")
            ax2.set_xlabel("Layers")
            ax3.set_xlabel("Layers")
            ax1.legend(labels=["Precision", "Recall", "F1"], loc='upper right')
            ax2.legend(labels=["Precision", "Recall", "F1"], loc='upper right')
            ax3.legend(labels=["Full", "120"], loc='upper right')
            fig1.show()
            plt.show()
            fig2.show()
            plt.show()
            fig3.show()
            plt.show()

import matplotlib.pyplot as plt


def plot_interaction_matrix(
    interactions, input_ids=None, sen=None, deps=None, save_file=None, add_ranks=False
):
    fig, ax = plt.subplots(figsize=(9, 9))
    ims = ax.imshow(
        interactions,
        cmap="coolwarm",
        clim=(-torch.max(abs(interactions)), torch.max(abs(interactions))),
    )
    # plt.colorbar(ims)

    if input_ids is not None:
        plt.xticks(range(len(input_ids)), input_ids.tolist())
        plt.yticks(range(len(input_ids)), input_ids.tolist())
    elif sen is not None:
        plt.xticks(range(len(sen)), sen, fontsize=12)
        plt.yticks(range(len(sen)), sen, fontsize=12)

    if deps is not None:
        for x, y in deps:
            rectangle = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                linewidth=3,
                facecolor="none",
                edgecolor="black",
                clip_on=False,
            )
            ax.add_patch(rectangle)

            rectangle = Rectangle(
                (y - 0.5, x - 0.5),
                1,
                1,
                linewidth=3,
                facecolor="none",
                edgecolor="black",
                clip_on=False,
            )
            ax.add_patch(rectangle)

    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["bottom"].set_color("none")

    plt.grid(False)

    if add_ranks:
        ax.text(6.5, -1, "Relative", fontsize=12, ha="center", fontweight="bold")
        ax.text(6.5, -0.7, "Rank", fontsize=12, ha="center", fontweight="bold")
        ax.text(6.5, 6, "ARR: 1.0", fontsize=12, ha="center", fontweight="bold")
        ax.text(2.5, -0.85, "Row Ranks", fontsize=12, ha="center", fontweight="bold")

        plt.plot([5.7, 7.3], [5.5, 5.5], color="0.5")

        for i, row in enumerate(interactions):
            ranks = torch.zeros_like(row).to(int)
            for idx, rank in enumerate(row.argsort()):
                ranks[rank] = idx

            for j, rank in enumerate(ranks.tolist()):
                ax.text(j, i, rank, fontsize=12, ha="center", va="center")

            ax.text(6.5, i + 0.1, "5/5", fontsize=12, ha="center")

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")

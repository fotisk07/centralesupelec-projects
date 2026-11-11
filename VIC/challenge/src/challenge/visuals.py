from matplotlib import patches

names = ["Hammer", "Knife", "Gun", "Wrench", "HandCuffs", "Bullet"]

names = {
    -1: "",
    0: "Hammer",
    1: "Knife",
    2: "Gun",
    3: "Wrench",
    4: "HandCuffs",
    5: "Bullet",
}


##### Utility functions used to plot images
# add yolo boxes should mostly be used with predictions
# and plot image when you want to visualise a sample
def add_yolo_boxes(ax, targets, img_shape, color="red", linewidth=2):
    if len(targets) == 0:
        return

    H, W = img_shape[:2]

    for cls, cx, cy, w, h in targets:
        x = (cx - w / 2) * W
        y = (cy - h / 2) * H
        bw = w * W
        bh = h * H
        rect = patches.Rectangle(
            (x, y), bw, bh, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(rect)

        ax.text(x, y - 3, names[cls], color=color, fontsize=11, verticalalignment="top")


def plot_image_with_boxes(ax, image, targets, **kwargs):
    ax.imshow(image)
    add_yolo_boxes(ax, targets, image.shape, **kwargs)

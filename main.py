import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stdin_arr = []
    x = []
    gt_y = []
    obs_y = []
    lstsq_y = []

    f = open(sys.argv[1], "r")
    for line in f:
        stdin_arr = line.split(",")

        length = len(stdin_arr)

        x.append(float(stdin_arr[0]))

        gt_y.append(float(stdin_arr[1]))

        obs_y.append(float(stdin_arr[2]))

        lstsq_y.append(float(stdin_arr[3]))

    f.close()

    plt.scatter(x, obs_y, color="g", marker="x")
    plt.plot(x, gt_y, color="b")
    plt.plot(x, lstsq_y, color="r", marker="o", fillstyle="none", linewidth=0)
    plt.legend(["ground truth","least squares fitted curve","observations"])
    plt.savefig("../assets/ceres-output.png")
    plt.show()

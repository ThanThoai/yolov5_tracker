import math
import numpy as np

def vertical_line(p1, p2):
    """
    Parameters
    ----------
    p1：判定线起始点, xy
    p2：判定线结束点, xy

    Returns
    -------
    画箭头的起始/结束坐标，和标志(out)
    """
    if (np.array(list(p1) + list(p2)) == 0).all():
        return (0, 0), (0, 0), None
    try:
        w, h = (p2[0] - p1[0]), (p2[1] - p1[1])
        # 找出判定线的中点与1/3的点，再基于中点对1/3的点方向 做顺时针旋转90度来指明out的方向
        p1_k = p1[0] + w / 2, p1[1] + h / 2
        p_v = p1[0] + w / 2.5, p1[1] + h / 2.5
        # p_v = p1[0] + w / 2 - 50, p1[1] + h / 2 - 50
        """
        基于数学中xy轴  逆时针旋转点公式修改;
        由于基于图像的坐标和数学中xy坐标不一致，y轴是反的；
        所以下面对应旋转之后的y坐标更改为(p1_k[1] - ..), 原公式为(p1_k[1] + ..)；
        且由于y轴反了，旋转角度也是反的，所以下面实则实现的是顺时针的旋转90度；
        """
        p2_k = (
            p1_k[0]
            + (p_v[0] - p1_k[0]) * math.cos(math.pi / 2)
            - (p_v[1] - p1_k[1]) * math.sin(math.pi / 2),
            p1_k[1]
            - (
                (p_v[1] - p1_k[1]) * math.cos(math.pi / 2)
                - (p_v[0] - p1_k[0]) * math.sin(math.pi / 2)
            ),
        )
        if vLineAngle((p1, p2), (p1_k, p2_k)) > 180:
            out = True
        else:
            out = False
        return np.int64(p1_k), np.int64(p2_k), out
    except:
        return (0, 0), (0, 0), None


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def vLineAngle(v1, v2):
    """
    计算两条直线的夹角(0-360)
    """
    dx1 = v1[1][0] - v1[0][0]
    dy1 = v1[1][1] - v1[0][1]
    dx2 = v2[1][0] - v2[0][0]
    dy2 = v2[1][1] - v2[0][1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = float(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = float(angle2 * 180 / math.pi)
    # print(angle2)
    included_angle = angle2 - angle1
    if included_angle < 0:
        included_angle += 360
    return included_angle


def getPointLen(p1, p2):
    """
    计算2点间距离
    """
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    return math.sqrt((x ** 2) + (y ** 2))


def point2LineDistance(point, line, abs=False):
    """
    计算点到线的距离
    abs: 取绝对值
    """
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0][0]
    line_s_y = line[0][1]
    line_e_x = line[1][0]
    line_e_y = line[1][1]
    # 若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x) if abs else point_x - line_s_x
    # 若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y) if abs else point_y - line_s_y
    # 斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    # 截距
    b = line_s_y - k * line_s_x
    # 带入公式得到距离dis
    t = math.fabs(k * point_x - point_y + b) if abs else k * point_x - point_y + b
    dis = t / math.pow(k * k + 1, 0.5)
    return dis
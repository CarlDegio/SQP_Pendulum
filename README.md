# SQP算法在pendulum的尝试
sqp_test.ipynb 测试了一个简单的情形，SQP可在这种情况下收敛
sqp_test_figure.py 同上，将图像绘制到外面
---
main.py 使用OSQP库直接解算，发现无法稳定收敛，并且推导比较复杂

cvx_main.py 使用CVXPY，可以比较简便的写出约束条件，但仍然因为原版SQP难以解决复杂约束而无法求解

scipy_trust.py 使用scipy的trust-constr进行有约束优化，效果比较好，但解算速度较慢
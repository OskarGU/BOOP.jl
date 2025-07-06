X_train = [1.0, 2.5, 4.0];  y_train = [sin(x) for x in X_train];
gp_model = GP(X_train', y_train, MeanZero(), SE(0.0, 0.0));
optimize!(gp_model);
# 2. Define the best observed value and a candidate point
y_best = minimum(y_train);
x_candidate = 3.0;
# 3. Compute Expected Improvement
ei = expected_improvement(gp_model, x_candidate, y_best; ξ=0.01)
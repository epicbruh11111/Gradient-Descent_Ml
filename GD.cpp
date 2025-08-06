#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

// Function to perform one step of gradient descent
std::pair<double, double> descent(const std::vector<double>& x,
    const std::vector<double>& y,
    double w,
    double b,
    double learning_rate) {
    double dldw = 0.0;
    double dldb = 0.0;
    size_t n = x.size();

    // Calculate gradients (scaled by 1/n during computation)
    for (size_t i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];
        double error = yi - (w * xi + b);
        dldw += -2.0 * xi * error / n;
        dldb += -2.0 * error / n;
    }

    // Update parameters
    w -= learning_rate * dldw;
    b -= learning_rate * dldb;

    return { w, b };
}

// Function to calculate mean squared error loss
double calculate_loss(const std::vector<double>& x,
    const std::vector<double>& y,
    double w,
    double b) {
    double loss = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double yhat = w * x[i] + b;
        loss += std::pow(y[i] - yhat, 2);
    }
    return loss / x.size();
}

// Main gradient descent function
void gradient_descent(const std::vector<double>& x,
    const std::vector<double>& y,
    double w = 0.0,
    double b = 0.0,
    double learning_rate = 0.01,
    size_t max_epochs = 400,
    double loss_threshold = 1e-6) {
    if (x.empty() || x.size() != y.size()) {
        std::cerr << "Error: Invalid input data (empty or mismatched sizes)." << std::endl;
        return;
    }

    double wf = w; // final weight
    double bf = b; // final bias
    double prev_loss = std::numeric_limits<double>::max();

    std::cout << std::fixed << std::setprecision(6);

    for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
        // Perform gradient descent step
        auto [new_w, new_b] = descent(x, y, wf, bf, learning_rate);
        wf = new_w;
        bf = new_b;

        // Calculate loss
        double loss = calculate_loss(x, y, wf, bf);

        // Print progress
        std::cout << "Epoch " << std::setw(4) << epoch
            << " loss: " << loss
            << ", parameters w: " << wf
            << ", b: " << bf << std::endl;

        // Check for convergence
        if (std::abs(prev_loss - loss) < loss_threshold) {
            std::cout << "Converged at epoch " << epoch << std::endl;
            break;
        }
        prev_loss = loss;
    }
}

// Example usage
int main() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, 0.1); // Reduced noise scale

    // Generate sample data
    std::vector<double> x(10);
    std::vector<double> y(10);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = norm_dist(gen);
        y[i] = 2.0 * x[i] + noise_dist(gen); // Linear relationship: y = 2x + noise
    }

    // Run gradient descent
    gradient_descent(x, y);

    return 0;
}

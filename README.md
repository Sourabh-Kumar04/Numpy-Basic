# NumPy-Basic

<div align="center">
  <img src="https://numpy.org/doc/stable/_static/numpylogo.svg" alt="NumPy Logo" width="400"/>
  <h3>A Comprehensive Guide to NumPy Fundamentals</h3>
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)](https://www.python.org/)
  [![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange)](https://numpy.org/)
  [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
  [![GitHub stars](https://img.shields.io/github/stars/Sourabh-Kumar04/Numpy-Basic?style=social)](https://github.com/Sourabh-Kumar04/Numpy-Basic/stargazers)
  
  <p>From zero to hero: A structured learning path for mastering NumPy</p>
  
  [Getting Started](#-getting-started) • 
  [Learning Path](#-learning-path) • 
  [Examples](#-code-examples) • 
  [Contributing](#-how-to-contribute) • 
  [License](#-license)
</div>

## 📋 Overview

This repository provides a **comprehensive, step-by-step guide** to mastering NumPy, the fundamental package for scientific computing in Python. Each phase builds systematically on previous knowledge, with practical examples and clear explanations.

### 🎯 Who Is This For?

- 🔰 **Beginners** looking to build a solid foundation in NumPy
- 🚀 **Intermediate users** wanting to deepen their understanding of advanced features
- 🎓 **Students** preparing for data science, machine learning, or AI coursework
- 💼 **Professionals** transitioning to roles requiring numerical computation skills

## ⚡ Quick Start

For those familiar with Python environments, get started immediately:

```bash
git clone https://github.com/Sourabh-Kumar04/Numpy-Basic.git
cd Numpy-Basic
python -m venv venv && source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Begin with `Phase_1/01_phase_1.ipynb` and progress through each phase sequentially.

## 📚 Learning Path

<div align="center">
  <img src="https://mermaid.ink/img/pako:eNp1kk1vgzAMhv9KlBOTOPQwCVpKdpgELPSwHvhQSIaqIYRkLpWG-O8LiQpDXQ6W_eT1h22QdpU0BJkJOQxlxWFn3d7YWukqVBqcg18MaIYDU2xVUsMjUCKZHqN51Ep48OG6Gw9Y4CjwZrIQPMgCi-VNRH8TqA33TgVaxN-ey1J0i_jD6Zx7zQlKTLd7U0qFnG3Ax-PxJigVKGuNhXvVusdHnTeLr4eLwOxnc4FXjCRPfUQPwqEQnJnXwXb6HQwU2ktrLKm_k-4WMBn-bZkWjqwOuq5CdYb9cQRXTukX9Xds0LlOlTSaBq4F5XfNZlGEF88o1WM5XEZ1ljGUEa71gMqZ27W7IWxAapDo9HWx6pA9ZymK3lSH9EIpx35ckadm0TblUwuXZEEFTb6wNxZlJN09XiZN?" alt="NumPy Learning Path" width="700"/>
</div>

### 🔍 Phase-by-Phase Progress

| Phase | Status | Topics | Key Concepts |
|-------|--------|--------|-------------|
| **🧩 Phase 1** | ✅ Complete | NumPy Fundamentals | Arrays vs Lists, Creating Arrays, Data Types, Basic Operations |
| **📍 Phase 2** | ✅ Complete | Data Manipulation | Indexing, Slicing, Sorting, Boolean Masks, Fancy Indexing |
| **🔄 Phase 3** | ✅ Complete | Array Transformation | Reshaping, Stacking, Splitting, Broadcasting Rules |
| **🧮 Phase 4** | 🚧 In Progress | Advanced Topics | Vector/Matrix Operations, Trigonometric Functions, Statistics, File Operations |

## 📁 Repository Structure

```
Numpy-Basic/
├── LICENSE                         # Apache 2.0 License
├── README.md                       # Project documentation
├── main.py                         # Example runner script
├── pyproject.toml                  # Project dependencies and configuration
├── uv.lock                         # Dependency lock file (for uv users)
├── Phase_1/
│   └── 01_phase_1.ipynb            # NumPy Basics: Arrays, Creation, Types
├── Phase_2/
│   └── 01_phase_2.ipynb            # Data Access: Indexing, Slicing, Filtering
├── Phase_3/
│   └── 01_phase_3.ipynb            # Data Transformation: Reshaping, Stacking, Broadcasting
└── Phase_4/
    ├── 01_phase_4.ipynb            # Advanced: Math Operations, Statistics, Visualization
    ├── array1.npy                  # Sample data file for practice
    ├── array2.npy                  # Sample data file for practice
    ├── array3.npy                  # Sample data file for practice
    └── numpy_logo.npy              # NumPy logo encoded as an array
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+ installed
- Git (for cloning the repository)
- Basic familiarity with Python programming

### Step-by-Step Setup

1. **Clone the repository**

```bash
git clone https://github.com/Sourabh-Kumar04/Numpy-Basic.git
cd Numpy-Basic
```

2. **Set up a virtual environment** (Choose your preferred method)

```bash
# Option 1: Standard venv
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Option 2: Using uv (faster alternative)
uv venv
uv activate
```

3. **Install dependencies**

```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using uv with pyproject.toml
uv pip install -e .
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

## 📖 What You'll Learn

### 🧩 Phase 1: NumPy Fundamentals

<div align="center">
  <img src="https://user-images.githubusercontent.com/1671563/154354937-2d910d39-2a43-4b10-b35a-9fd7097bcb15.png" alt="NumPy Array Illustration" width="400"/>
</div>

- Why NumPy over standard Python lists?
  - Performance benchmarks showing speed differences
  - Memory efficiency comparisons
  - Vectorized operations
- Creating arrays from different sources
  - From Python lists
  - Using built-in functions: `zeros()`, `ones()`, `arange()`, `linspace()`
  - Random number generation
- Understanding array data types and properties

### 📍 Phase 2: Data Manipulation

- Accessing array elements
  - Basic indexing vs fancy indexing
  - Difference between views and copies
- Slicing multi-dimensional arrays
- Advanced selection with boolean masks
  - Filtering data with conditions
  - Combining multiple conditions
- Practical comparison between `np.where()` and boolean indexing
- Sorting arrays and finding unique values

### 🔄 Phase 3: Array Transformation

- Inspecting array properties
  - Shape, size, dimensions, data type
- Reshaping arrays
  - `reshape()`, `ravel()`, `flatten()`
  - Adding/removing dimensions with `newaxis` and `squeeze()`
- Combining arrays
  - Vertical stacking with `vstack()`
  - Horizontal stacking with `hstack()`
  - General stacking with `concatenate()`
- Broadcasting rules and compatibility
  - When operations work between arrays of different shapes
  - Common broadcasting errors and how to fix them

### 🧮 Phase 4: Applications & Advanced Features

- Vector, matrix, and tensor operations
  - Dot products, cross products, matrix multiplication
  - Linear algebra operations
- Comprehensive angle function reference
  - Trigonometric functions (`sin`, `cos`, `tan`)
  - Inverse trigonometric functions (`arcsin`, `arccos`, `arctan2`)
- Statistical functions for data analysis
  - Measures of central tendency
  - Measures of dispersion
  - Percentiles and quantiles
- Working with NumPy's native file formats
  - `.npy` for single arrays
  - `.npz` for multiple arrays
- Data visualization with matplotlib

## 📊 Code Examples

### Performance Comparison: Lists vs. NumPy Arrays

```python
import numpy as np
import time

# Python list operation
start = time.time()
python_list = list(range(1000000))
python_list = [x * 2 for x in python_list]
list_time = time.time() - start

# NumPy array operation
start = time.time()
numpy_array = np.arange(1000000)
numpy_array = numpy_array * 2
numpy_time = time.time() - start

print(f"Python list processing time: {list_time:.5f} seconds")
print(f"NumPy array processing time: {numpy_time:.5f} seconds")
print(f"NumPy is {list_time/numpy_time:.1f}x faster!")
```

<details>
<summary>Output</summary>

```
Python list processing time: 0.12345 seconds
NumPy array processing time: 0.00567 seconds
NumPy is 21.8x faster!
```
</details>

### Fancy Indexing & Masking

```python
import numpy as np

# Create sample data
data = np.random.randint(0, 100, size=(5, 5))
print("Original data:")
print(data)

# Boolean masking (values greater than 50)
mask = data > 50
filtered_data = data[mask]
print("\nValues greater than 50:")
print(filtered_data)

# Using np.where() for conditional values
result = np.where(data > 50, data * 2, data)
print("\nValues > 50 doubled, others unchanged:")
print(result)
```

### Visualization with Dark Mode

```python
import matplotlib.pyplot as plt
import numpy as np

# Set dark style
plt.style.use('dark_background')

# Generate data
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='cyan', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='magenta', linewidth=2)
plt.title("Trigonometric Functions", fontsize=16)
plt.xlabel("x (radians)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

<div align="center">
  <img src="https://user-images.githubusercontent.com/1671563/157742658-5a197551-29a4-4f56-8e6c-ed0146fa9dc3.png" alt="Dark Mode Visualization Example" width="600"/>
</div>

## 📈 Statistical Functions Reference

| Function | Description | Example |
|----------|-------------|---------|
| `np.mean()` | Arithmetic mean | `np.mean(arr, axis=0)` |
| `np.median()` | Median value | `np.median(arr)` |
| `np.std()` | Standard deviation | `np.std(arr, ddof=1)` |
| `np.var()` | Variance | `np.var(arr)` |
| `np.min()` | Minimum value | `np.min(arr, axis=1)` |
| `np.max()` | Maximum value | `np.max(arr)` |
| `np.percentile()` | nth percentile | `np.percentile(arr, 75)` |
| `np.quantile()` | nth quantile | `np.quantile(arr, [0.25, 0.5, 0.75])` |
| `np.corrcoef()` | Correlation coefficient | `np.corrcoef(x, y)` |
| `np.cov()` | Covariance matrix | `np.cov(x, y)` |

## 💾 Working with .npy Files

```python
import numpy as np

# Create sample array
array = np.random.normal(0, 1, size=(100, 100))

# Save to .npy file
np.save('sample_array.npy', array)

# Load from .npy file
loaded_array = np.load('sample_array.npy')

# Verify it's the same
print("Arrays are identical:", np.array_equal(array, loaded_array))
```

## 🤔 Common Questions & Answers

<details>
<summary><b>Why use NumPy instead of Python lists?</b></summary>
NumPy arrays are more efficient than Python lists for numerical operations because:
<ul>
  <li>They store data in contiguous memory blocks</li>
  <li>They leverage vectorized operations (SIMD instructions)</li>
  <li>They offer specialized numerical functions optimized in C</li>
  <li>They use less memory for the same amount of numerical data</li>
</ul>
</details>

<details>
<summary><b>What's the difference between a view and a copy?</b></summary>
<ul>
  <li>A <b>view</b> is just a different way to access the same data - changes to the view affect the original array</li>
  <li>A <b>copy</b> is a new array with the same values - changes to the copy don't affect the original</li>
  <li>Basic slicing typically returns views, while advanced indexing returns copies</li>
</ul>
</details>

<details>
<summary><b>What are broadcasting rules?</b></summary>
Broadcasting allows NumPy to perform operations on arrays of different shapes. The rules are:
<ul>
  <li>Arrays are compared from their trailing dimensions</li>
  <li>Dimensions with size 1 are stretched to match the other array</li>
  <li>Missing dimensions are treated as having size 1</li>
  <li>If dimensions are compatible, broadcasting proceeds</li>
</ul>
</details>

## 📚 Additional Resources

- [Official NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy Cheat Sheet (PDF)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
- [NumPy Tutorials](https://numpy.org/numpy-tutorials/)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)

## 🧠 How to Contribute

Contributions to improve this repository are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create a branch** for your feature or fix
3. **Commit** your changes with descriptive messages
4. **Push** to your branch
5. Submit a **Pull Request**

### Commit Message Convention

```
git commit -m "✨ Added Phase_3: Array reshaping and broadcasting examples"
```

| Emoji | Description |
|-------|-------------|
| ✨ | New features or content |
| 🐛 | Bug fixes |
| 📝 | Documentation updates |
| 🔧 | Configuration changes |
| 🧹 | Code cleanup |
| 🎨 | Style improvements |

## 💬 Community & Support

- **GitHub Discussions**: [Open a discussion](https://github.com/Sourabh-Kumar04/Numpy-Basic/discussions)
- **Issue Tracker**: [Report bugs or request features](https://github.com/Sourabh-Kumar04/Numpy-Basic/issues)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 📊 Citation

If you use this repository in your research or educational materials, please cite it as:

```bibtex
@misc{kumar2025numpybasic,
  author = {Kumar, Sourabh},
  title = {NumPy-Basic: Comprehensive Guide to NumPy Fundamentals},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sourabh-Kumar04/Numpy-Basic},
  howpublished = {\url{https://github.com/Sourabh-Kumar04/Numpy-Basic}},
}
```

## 🌐 Connect & Support

- **Author**: Sourabh Kumar
- **GitHub**: [@Sourabh-Kumar04](https://github.com/Sourabh-Kumar04)
- **LinkedIn**: [linkedin.com/in/sourabh-kumar04](https://linkedin.com/in/sourabh-kumar04)

---

<div align="center">
  <p>If you find this repository helpful, please consider starring it! ⭐️</p>
  <p>
    <a href="https://www.buymeacoffee.com/YourUsername" target="_blank">
      <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="40px">
    </a>
  </p>
  <p>"NumPy doesn't just compute numbers—it transforms how we think about data." 🧮</p>
</div>

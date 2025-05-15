# 📊 Numpy-Basic: Comprehensive Guide to NumPy Fundamentals

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

A structured, beginner-to-intermediate learning path for mastering NumPy, the foundation of scientific computing in Python.

## 🎯 Who Is This For?

- 🚀 **Beginners** looking to build a solid foundation in NumPy
- 👨‍💻 **Intermediate users** wanting to deepen their understanding of advanced features
- 🎓 **Students** preparing for data science, machine learning, or AI coursework
- 💼 **Professionals** transitioning to roles requiring numerical computation skills

## 📚 What You'll Learn

| Phase | Topics | Key Concepts |
|-------|--------|-------------|
| **🧩 Phase 1** | NumPy Fundamentals | Arrays vs Lists, Creating Arrays, Data Types, Basic Operations |
| **📍 Phase 2** | Data Manipulation | Indexing, Slicing, Sorting, Boolean Masks, Fancy Indexing |
| **🔄 Phase 3** | Array Transformation | Reshaping, Stacking, Splitting, Broadcasting Rules |
| **🧮 Phase 4** | Applications & Advanced | Vector/Matrix Operations, Trigonometric Functions, Statistics, File Operations |
| **📈 Bonus** | Visualization | Matplotlib Integration, Customization, Themes |

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
- Basic familiarity with Python

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

## 🔍 Phase-by-Phase Overview

### 🧩 Phase 1: NumPy Fundamentals

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

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🌐 Connect & Support

- **Author**: Sourabh Kumar
- **GitHub**: [@Sourabh-Kumar04](https://github.com/Sourabh-Kumar04)
- **LinkedIn**: [linkedin.com/in/sourabh-kumar04](https://linkedin.com/in/sourabh-kumar04)

---

> "NumPy doesn't just compute numbers—it transforms how we think about data." 🧮

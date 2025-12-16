# Contributing to Hierarchical Diffusion

This is primarily an **educational framework** designed for self-directed learning. The goal is to help students understand diffusion priors and mode collapse through hands-on implementation, not to be a production-ready research codebase.

## 🤝 How to Contribute

We welcome contributions!

### ✅ Encouraged Contributions

- **Bug fixes** - Typos, logic errors, broken imports
- **Documentation improvements** - Clarifications, better explanations, fixed examples
- **Extended evaluation metrics** - Additional ways to measure mode collapse
- **Visualization enhancements** - Better plots, comparative visualizations
- **Installation/setup fixes** - Compatibility issues, missing dependencies
- **Performance optimizations** - Faster training without changing learning objectives
- **Python extras for other frameworks** - e.g. JAX support
- **Additional educational examples** - New synthetic datasets, extended experiments

### ⚠️ Please Discuss First

- Major architectural changes
- New model variants (Model C, etc.)
- Significant API changes
- Changes that would affect the learning path

## 📝 Contribution Process

1. **Open an Issue** first to discuss your idea
2. **Fork the repository**
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** with clear commit messages
5. **Test your changes**: Ensure `train.py` and `evaluate.py` still work
6. **Update documentation** if needed
7. **Submit a Pull Request** with a clear description

## 🧪 Testing Your Changes

Before submitting:

```bash
# Test dataset generation
python -m data.synthetic_dataset

# Test training (small run)
python train.py --model flat --epochs 2 --batch-size 8

# Test evaluation
python evaluate.py --model flat --checkpoint outputs/flat_final.pt
```

## 📐 Code Style

- Follow existing code style (Black)
- Add docstrings to new functions/classes
- Include type hints where helpful
- Keep educational clarity as the top priority

## 🎓 Educational Standards

When contributing:

- **Maintain clarity**: Code should be understandable to ML students
- **Preserve learning objectives**: Don't "solve" the implementation tasks for students
- **Document thoroughly**: Explain *why*, not just *what*
- **Keep it minimal**: Avoid unnecessary complexity

## 📊 Documentation Changes

- **README.md** - High-level overview and quick start
- **QUICKSTART.md** - Week-by-week student plan
- **ASSIGNMENT.md** - Detailed implementation instructions
- **SETUP.md** - Installation and usage
- **FRAMEWORK_SUMMARY.md** - Technical deep dive

Ensure any changes are reflected in the relevant docs.

## 🐛 Reporting Bugs

When filing an issue, include:

- Python version
- PyTorch version
- Operating system
- Full error message and stack trace
- Minimal code to reproduce
- What you expected vs. what happened

## 💡 Suggesting Features

For feature requests:

- Explain the educational value
- Describe the use case
- Consider implementation complexity
- Note if you're willing to implement it

## 📄 License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.

## ❓ Questions?

Open an issue with the label `question` - we're happy to help!

---

Thank you for helping make this a better learning resource! 🙏

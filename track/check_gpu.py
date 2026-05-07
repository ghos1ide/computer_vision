from runtime import import_jittor_or_exit, print_startup_hint


def main():
    print("=" * 40)
    print(" Jittor GPU Environment Check ")
    print("=" * 40)

    print_startup_hint("GPU environment check")
    jt, _ = import_jittor_or_exit("GPU environment check")

    has_cuda = jt.has_cuda
    print(f"CUDA Available (jt.has_cuda): {has_cuda}")

    if has_cuda:
        try:
            jt.flags.use_cuda = 1
            print("Successfully enabled CUDA (jt.flags.use_cuda = 1).")

            a = jt.ones((2, 2))
            b = a + a
            b.sync()
            print("GPU computation test passed successfully!")
        except Exception as exc:
            print(f"Error during GPU operation: {exc}")
    else:
        print("CUDA is NOT available. Jittor will fall back to CPU.")
        print("Please check NVIDIA driver, CUDA toolkit, and Jittor setup.")

    print("=" * 40)


if __name__ == "__main__":
    main()

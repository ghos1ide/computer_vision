import jittor as jt

def main():
    print("="*40)
    print(" Jittor GPU Environment Check ")
    print("="*40)
    
    # 检查是否支持 CUDA
    has_cuda = jt.has_cuda
    print(f"CUDA Available (jt.has_cuda): {has_cuda}")
    
    if has_cuda:
        try:
            # 尝试开启 CUDA
            jt.flags.use_cuda = 1
            print("Successfully enabled CUDA (jt.flags.use_cuda = 1).")
            
            # 创建一个张量并进行简单运算以确认 GPU 能正常工作
            a = jt.ones((2, 2))
            b = a + a
            b.sync()
            print("GPU computation test passed successfully!")
        except Exception as e:
            print(f"Error during GPU operation: {e}")
    else:
        print("CUDA is NOT available. Jittor will fall back to CPU.")
        print("Please check your NVIDIA drivers, CUDA tookit installation, and Jittor setup.")
    print("="*40)

if __name__ == "__main__":
    main()

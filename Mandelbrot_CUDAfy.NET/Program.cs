using System;
using System.Drawing;
using System.Drawing.Imaging;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;


namespace Mandelbrot_CUDAfy.NET
{
    class Program
    {
        public static void Main()
        {
            // PRZYGOTOWANIE
            
            // wstęp dla cudafy.net
            Console.WriteLine("CUDAfy.NET / C# / Mandelbrot with CUDA GPU\n\n");
            PrintGpuProperties();
            CudafyModes.Target = eGPUType.Cuda;
            CudafyModes.DeviceId = 0;
            CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;


            // sprawdź wszystkie dostępne urządzenia
            if (CudafyHost.GetDeviceCount(CudafyModes.Target) == 0)
                throw new System.ArgumentException("Nie znaleziono odpowiedniego urządzenia", "original");

            // inicjalizacja urządzenia i wypisanie właściwości
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            CudafyModule km = CudafyTranslator.Cudafy(); // daj moduł dla GPU
            gpu.LoadModule(km);



            // dane do GPU
            const int pictureWidth = 2048, pictureHeight = 2048;
            double[,] bmArrayCPU = new double[pictureWidth, pictureHeight];
            double[] bmArrayGPU = new double[pictureWidth * pictureHeight];
            double[] dev_bmArrayGPU = gpu.Allocate<double>(bmArrayGPU); ;
            Bitmap bmCPU = new Bitmap(pictureWidth, pictureHeight);
            Bitmap bmGPU = new Bitmap(pictureWidth, pictureHeight);



            // CPU starts
            // liczę czas wykonania metody/funkcji
            var watch_CPU = System.Diagnostics.Stopwatch.StartNew();
            mandelbrot_CPU(pictureWidth, pictureHeight, bmArrayCPU);
            for (int x = 0; x < pictureWidth; x++)
            {
                for (int y = 0; y < pictureHeight; y++)
                {
                    bmCPU.SetPixel(x, y, bmArrayCPU[x, y] == 1 ? Color.Black : Color.White);
                }
            }
            bmCPU.Save("outputMandelbrot_CPU.jpg", ImageFormat.Jpeg);
            watch_CPU.Stop();
            var elapsedMs_CPU = watch_CPU.ElapsedMilliseconds;
            Console.WriteLine("-- -- --  -- -- --");
            Console.WriteLine("-- CPU: {0} ms --", elapsedMs_CPU);
            // CPU end


            // GPU starts
            dim3 threadsPerBlock = new dim3(8, 8);
            dim3 numBlocks = new dim3(pictureWidth / threadsPerBlock.x, pictureHeight / threadsPerBlock.y);

            var watch_GPU = System.Diagnostics.Stopwatch.StartNew();
            gpu.Launch(numBlocks, threadsPerBlock).mandelbrot_GPU(pictureWidth, pictureHeight, dev_bmArrayGPU);
            gpu.CopyFromDevice(dev_bmArrayGPU, bmArrayGPU);

            for (int x = 0; x < pictureWidth; x++)
            {
                for (int y = 0; y < pictureHeight; y++)
                {
                    bmGPU.SetPixel(x, y, bmArrayGPU[pictureHeight * x + y] == 1 ? Color.Black : Color.White);
                }
            }
            bmGPU.Save("outputMandelbrot_GPU.jpg", ImageFormat.Jpeg);
            watch_GPU.Stop();
            var elapsedMs_GPU = watch_GPU.ElapsedMilliseconds;
            Console.WriteLine("-- GPU: {0} ms --", elapsedMs_GPU);
            Console.WriteLine("-- -- --  -- -- --");
            // GPU end


            Console.WriteLine("Koniec!");
            Console.ReadKey();
        }


        
        public static void mandelbrot_CPU(int width, int height, double[,] dev_bmArray)
        {
            int max = 1000;
            for (int row = 0; row < height; row++)
            {
                for (int col = 0; col < width; col++)
                {
                    double c_re = (col - width / 2) * 4.0 / width;
                    double c_im = (row - height / 2) * 4.0 / width;
                    double x = 0, y = 0;
                    int iterations = 0;
                    while (x * x + y * y < 4 && iterations < max)
                    {
                        double x_new = x * x - y * y + c_re;
                        y = 2 * x * y + c_im;
                        x = x_new;
                        iterations++;
                    }
                    if (iterations < max) dev_bmArray[col, row] = 0;
                    else dev_bmArray[col, row] = 1;
                }
            }
        }

        [Cudafy]
        public static void mandelbrot_GPU(GThread thread, int width, int height, double[] dev_bmArray)
        {
            int max = 1000;
            var threadx = (thread.blockDim.x * thread.blockIdx.x) + thread.threadIdx.x;
            var thready = (thread.blockDim.y * thread.blockIdx.y) + thread.threadIdx.y;

            double c_re = (threadx - width / 2) * 4.0 / width;
            double c_im = (thready - height / 2) * 4.0 / width;
            double x = 0, y = 0;
            int iterations = 0;
            while (x * x + y * y < 4 && iterations < max)
            {
                double x_new = x * x - y * y + c_re;
                y = 2 * x * y + c_im;
                x = x_new;
                iterations++;
            }
            if (iterations < max) dev_bmArray[width * threadx + thready] = 0;
            else dev_bmArray[width * threadx + thready] = 1;

        }

        // WŁAŚCIWOŚCI GPU
        public static void PrintGpuProperties() // CUDAfy properties sample
        {
            int i = 0;
            foreach (GPGPUProperties devicePropsContainer in CudafyHost.GetDeviceProperties(CudafyModes.Target, false))
            {
                Console.WriteLine("   --- General Information for device {0} ---", i);
                Console.WriteLine("Name:  {0}", devicePropsContainer.Name);
                Console.WriteLine("Platform Name:  {0}", devicePropsContainer.PlatformName);
                Console.WriteLine("Device Id:  {0}", devicePropsContainer.DeviceId);
                Console.WriteLine("Compute capability:  {0}.{1}", devicePropsContainer.Capability.Major, devicePropsContainer.Capability.Minor);
                Console.WriteLine("Clock rate: {0}", devicePropsContainer.ClockRate);
                Console.WriteLine("Simulated: {0}", devicePropsContainer.IsSimulated);
                Console.WriteLine();

                Console.WriteLine("   --- Memory Information for device {0} ---", i);
                Console.WriteLine("Total global mem:  {0}", devicePropsContainer.TotalMemory);
                Console.WriteLine("Total constant Mem:  {0}", devicePropsContainer.TotalConstantMemory);
                Console.WriteLine("Max mem pitch:  {0}", devicePropsContainer.MemoryPitch);
                Console.WriteLine("Texture Alignment:  {0}", devicePropsContainer.TextureAlignment);
                Console.WriteLine();

                Console.WriteLine("   --- MP Information for device {0} ---", i);
                Console.WriteLine("Shared mem per mp: {0}", devicePropsContainer.SharedMemoryPerBlock);
                Console.WriteLine("Registers per mp:  {0}", devicePropsContainer.RegistersPerBlock);
                Console.WriteLine("Threads in warp:  {0}", devicePropsContainer.WarpSize);
                Console.WriteLine("Max threads per block:  {0}", devicePropsContainer.MaxThreadsPerBlock);
                Console.WriteLine("Max thread dimensions:  ({0}(x), {1}(y), {2}(z))", devicePropsContainer.MaxThreadsSize.x, devicePropsContainer.MaxThreadsSize.y, devicePropsContainer.MaxThreadsSize.z);
                Console.WriteLine("Max grid dimensions:  ({0}(x), {1}(y), {2}(z))", devicePropsContainer.MaxGridSize.x, devicePropsContainer.MaxGridSize.y, devicePropsContainer.MaxGridSize.z);

                Console.WriteLine();

                i++;
            }
        }
    }
}

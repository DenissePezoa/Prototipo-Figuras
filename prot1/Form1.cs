using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Cvb;
using Emgu.CV.Cuda;
//using Emgu.CV.UI;
using Emgu.CV.VideoStab;


namespace prot1
{
    public partial class Form1 : Form
    {
        Image<Bgr, Byte> imge;
        VideoCapture _capture;
        private Mat _frame;
        private const int Threshold = 1;
        private const int ErodeIterations = 1;
        private const int DilateIterations = 7;
        private static MCvScalar drawingColor = new Bgr(Color.Blue).MCvScalar;

        private async void ProcessFrame(object sender, EventArgs e)
        {
            if (_capture != null && _capture.Ptr != IntPtr.Zero)
            {
                _capture.Retrieve(_frame, 0);
                pictureBox1.Image = _frame.Bitmap;
                double fps = 15;
                await Task.Delay(1000 / Convert.ToInt32(fps));

            }
        }
        public Form1()
        {
            InitializeComponent();
            _capture = new VideoCapture(1);


            _capture.ImageGrabbed += ProcessFrame;
            _frame = new Mat();
            if (_capture != null)
            {
                try
                {
                    _capture.Start();
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
        }

        private async void  button1_Click(object sender, EventArgs e)
        {

            if (_frame.IsEmpty)
            {
                return;
            }

            //try
            //{

                Mat m = new Mat();
                Mat n = new Mat();
                Mat o = new Mat();
                Mat binaryDiffFrame = new Mat();
                Mat denoisedDiffFrame = new Mat();
                Mat finalFrame = new Mat();
                Rectangle cropbox = new Rectangle();


                //
                //OBTENER COLOR ROJO
                //
                imge = _frame.ToImage<Bgr, Byte>();
                //Transformar a espacio de color HSV
                Image<Hsv, Byte> hsvimg = imge.Convert<Hsv, Byte>();

                //extract the hue and value channels
                Image<Gray, Byte>[] channels = hsvimg.Split();  //separar en componentes
                Image<Gray, Byte> imghue = channels[0];            //hsv, channels[0] es hue.
                Image<Gray, Byte> imgval = channels[2];            //hsv, channels[2] es value.

                //Filtro color
                //140 en adelante
                Image<Gray, byte> huefilter = imghue.InRange(new Gray(140), new Gray(255));
                //Filtro colores menos brillantes
                Image<Gray, byte> valfilter = imgval.InRange(new Gray(100), new Gray(255));
                //Filtro de saturación - quitar blancos 
                channels[1]._ThresholdBinary(new Gray(20), new Gray(255)); // Saturacion

                //Unir los filtros para obtener la imagen
                Image<Gray, byte> colordetimg = huefilter.And(valfilter).And(channels[1]);//aqui habia un Not()

                //Colorear imagen
                Image<Bgr, byte> ret = imge;
                var mat = imge.Mat;
                mat.SetTo(new MCvScalar(0, 0, 255), colordetimg);
                mat.CopyTo(ret);
                //Image<Bgr, byte> imgout = ret.CopyBlank();//imagen sin fondo negro
               
                ret._Or(imge);
                //Muestra imagen con los rojos destacados
                pictureBox2.Image = ret.Bitmap;


                //
                //COMIENZA OBTENCION DE BORDES
                //
                CvInvoke.AbsDiff(_frame, ret, n);

                CvInvoke.CvtColor(colordetimg, o, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(colordetimg, binaryDiffFrame, 5, 255, ThresholdType.Binary);


                CvInvoke.Erode(binaryDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), ErodeIterations, BorderType.Default, new MCvScalar(1));
                CvInvoke.Dilate(denoisedDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), DilateIterations, BorderType.Default, new MCvScalar(1));
                //pictureBox3.Image = denoisedDiffFrame.Bitmap;

                _frame.CopyTo(finalFrame);

                DetectObject(denoisedDiffFrame, finalFrame, cropbox);


                pictureBox3.Image = finalFrame.Bitmap;

                double fps = 15;
                await Task.Delay(1000 / Convert.ToInt32(fps));


           /* }
            catch (Exception ex)
            {
              MessageBox.Show(ex.Message);

             }

                */
        }







        //  
        //COMIENZAN FUNCIONES DE EDDIE
        //
        private void DetectObject(Mat detectionFrame, Mat displayFrame, Rectangle box)
        {
            Image<Bgr, Byte> buffer_im = displayFrame.ToImage<Bgr, Byte>();
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                VectorOfPoint biggestContour = null;
                IOutputArray hirarchy = null;

                CvInvoke.FindContours(detectionFrame, contours, hirarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);


                if (contours.Size > 0)
                {
                    double maxArea = 0;
                    int chosen = 0;
                    VectorOfPoint contour = null;
                    for (int i = 0; i < contours.Size; i++)
                    {
                        contour = contours[i];

                        double area = CvInvoke.ContourArea(contour);
                        if (area > maxArea)
                        {
                            maxArea = area;
                            chosen = i;
                        }
                    }


                    //MarkDetectedObject(displayFrame, contours[chosen], maxArea);//dibuja una envoltura roja

                    VectorOfPoint hullPoints = new VectorOfPoint();
                    VectorOfInt hullInt = new VectorOfInt();

                    CvInvoke.ConvexHull(contours[chosen], hullPoints, true);
                    CvInvoke.ConvexHull(contours[chosen], hullInt, false);

                    Mat defects = new Mat();


                    if (hullInt.Size > 3)
                        CvInvoke.ConvexityDefects(contours[chosen], hullInt, defects);

                    box = CvInvoke.BoundingRectangle(hullPoints);
                    CvInvoke.Rectangle(displayFrame, box, drawingColor);//Box rectangulo que encierra el area mas grande
                                                                        // cropbox = crop_color_frame(displayFrame, box);

                    buffer_im.ROI = box;

                    Image<Bgr, Byte> cropped_im = buffer_im.Copy();
                    //pictureBox8.Image = cropped_im.Bitmap;
                    Point center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);//centro  rectangulo MOUSE

                    VectorOfPoint start_points = new VectorOfPoint();
                    VectorOfPoint far_points = new VectorOfPoint();

                    if (!defects.IsEmpty)
                    {

                        Matrix<int> m = new Matrix<int>(defects.Rows, defects.Cols,
                           defects.NumberOfChannels);
                        defects.CopyTo(m);
                        int xe = 2000, ye = 2000;
                        int xs = 2000, ys = 2000;
                        int xer = 2000, yer = 2000;
                        int xsr = 2000, ysr = 2000;
                        int xem = 0, yem = 0;
                        int xsm = 0, ysm = 0;
                        int xez = 0, yez = 0;
                        int xsz = 0, ysz = 0;
                        int y = 0, x = 0;
                        int ym = 0, xm = 0;
                        int yr = 0, xr = 0;
                        int yz = 0, xz = 0;
                        for (int i = 0; i < m.Rows; i++)
                        {
                            int startIdx = m.Data[i, 0];
                            int endIdx = m.Data[i, 1];
                            int farIdx = m.Data[i, 2];
                            Point startPoint = contours[chosen][startIdx];
                            Point endPoint = contours[chosen][endIdx];
                            Point farPoint = contours[chosen][farIdx];
                            CvInvoke.Circle(displayFrame, endPoint, 3, new MCvScalar(0, 255, 255));
                            CvInvoke.Circle(displayFrame, startPoint, 3, new MCvScalar(255, 255, 0));

                            if (true)
                            {
                                if (endPoint.Y < ye)
                                {
                                    xe = endPoint.X;

                                    ye = endPoint.Y;

                                }

                                if (startPoint.Y < ys)
                                {
                                    xs = startPoint.X;

                                    ys = startPoint.Y;


                                }

                                if (ye < ys)
                                {
                                    y = ye;
                                    x = xe;



                                }
                                else
                                {
                                    y = ys;
                                    x = xs;
                                }


                                if (endPoint.Y > yem)
                                {
                                    xem = endPoint.X;

                                    yem = endPoint.Y;

                                }

                                if (startPoint.Y > ysm)
                                {
                                    xsm = startPoint.X;

                                    ysm = startPoint.Y;


                                }

                                if (yem > ysm)
                                {
                                    ym = yem;
                                    xm = xem;



                                }
                                else
                                {
                                    y = ys;
                                    x = xs;
                                }

                                if (endPoint.X < xer)
                                {
                                    xer = endPoint.X;

                                    yer = endPoint.Y;

                                }

                                if (startPoint.X < xsr)
                                {
                                    xsr = startPoint.X;

                                    ysr = startPoint.Y;


                                }

                                if (xer < xsr)
                                {
                                    yr = yer;
                                    xr = xer;



                                }
                                else
                                {
                                    yr = ysr;
                                    xr = xsr;
                                }


                                if (endPoint.X > xez)
                                {
                                    xez = endPoint.X;

                                    yez = endPoint.Y;

                                }

                                if (startPoint.X > xsz)
                                {
                                    xsz = startPoint.X;

                                    ysz = startPoint.Y;


                                }

                                if (xez > xsz)
                                {
                                    yz = yez;
                                    xz = xez;



                                }

                                else
                                {
                                    yz = ysz;
                                    xz = xsz;



                                }


                            }
                            /*var info = new string[] {
                
                $"Posicion: {endPoint.X}, {endPoint.Y}"
            };
                            WriteMultilineText(displayFrame, info, new Point(endPoint.X + 5, endPoint.Y));*/

                            double distance = Math.Round(Math.Sqrt(Math.Pow((center.X - farPoint.X), 2) + Math.Pow((center.Y - farPoint.Y), 2)), 1);
                            if (distance < box.Height * 0.3)
                            {
                                CvInvoke.Circle(displayFrame, farPoint, 3, new MCvScalar(255, 0, 0));
                            }

                            CvInvoke.Line(displayFrame, startPoint, endPoint, new MCvScalar(0, 255, 0));
                            // CvInvoke.Line(displayFrame, startPoint, farPoint, new MCvScalar(0, 255, 255));
                        }
                        var infoe = new string[] { $"Punto", $"Posicion: {x}, {y}" };
                        var infos = new string[] { $"Punto", $"Posicion: {xm}, {ym}" };
                        var infor = new string[] { $"Punto", $"Posicion: {x}, {y}" };
                        var infoz = new string[] { $"Punto", $"Posicion: {xm}, {ym}" };
                        var infoCentro = new string[] { $"Centro", $"Posicion: {xm}, {ym}" };

                        var xCentro = (x + xm + xr + xz) / 4;
                        var yCentro = (y + ym + yr + yz) / 4;

                        WriteMultilineText(displayFrame, infoe, new Point(x + 30, y));
                        CvInvoke.Circle(displayFrame, new Point(x, y), 5, new MCvScalar(255, 0, 255), 2);
                        Image<Bgr, byte> temp = detectionFrame.ToImage<Bgr, byte>();
                        var temp2 = temp.SmoothGaussian(5).Convert<Gray, byte>().ThresholdBinary(new Gray(230), new Gray(255));
                        VectorOfVectorOfPoint contorno = new VectorOfVectorOfPoint();
                        Mat mat = new Mat();
                        CvInvoke.FindContours(temp2, contorno, mat, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                        for (int i = 0; i < contorno.Size; i++)
                        {
                            double perimetro = CvInvoke.ArcLength(contorno[i], true);
                            VectorOfPoint approx = new VectorOfPoint();
                            CvInvoke.ApproxPolyDP(contorno[i], approx, 0.04 * perimetro, true);
                            CvInvoke.DrawContours(displayFrame, contorno, i, new MCvScalar(0, 255, 255), 2);

                        }

                        WriteMultilineText(displayFrame, infos, new Point(xm + 30, ym));
                        CvInvoke.Circle(displayFrame, new Point(xm, ym), 5, new MCvScalar(255, 0, 255), 2);
                        WriteMultilineText(displayFrame, infor, new Point(xr + 30, yr));
                        CvInvoke.Circle(displayFrame, new Point(xr, yr), 5, new MCvScalar(255, 0, 255), 2);
                        WriteMultilineText(displayFrame, infoz, new Point(xz + 30, yz));
                        CvInvoke.Circle(displayFrame, new Point(xz, yz), 5, new MCvScalar(255, 0, 255), 2);

                        WriteMultilineText(displayFrame, infoz, new Point(xCentro + 30, yCentro));
                        CvInvoke.Circle(displayFrame, new Point(xCentro, yCentro), 2, new MCvScalar(0, 100, 0), 4);
                        //CvInvoke.Circle(picture, new Point(x * 2, y * 4), 20, new MCvScalar(255, 0, 255), 2);*/

                    }

                }

            }

        }



        private static void WriteMultilineText(Mat frame, string[] lines, Point origin)
        {
            for (int i = 0; i < lines.Length; i++)
            {
                int y = i * 10 + origin.Y;
                CvInvoke.PutText(frame, lines[i], new Point(origin.X, y), FontFace.HersheyPlain, 0.8, drawingColor);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (_frame.IsEmpty)
            {
                return;
            }

            Image<Bgr, byte> img = _frame.ToImage<Bgr, byte>();
            //Transformar a espacio de color HSV
            Image<Hsv, Byte> hsvimg = img.Convert<Hsv, Byte>();

            //extract the hue and value channels
            Image<Gray, Byte>[] channels = hsvimg.Split();  //separar en componentes
            Image<Gray, Byte> imghue = channels[0];            //hsv, channels[0] es hue.
            Image<Gray, Byte> imgval = channels[2];            //hsv, channels[2] es value.

            //Filtro color
            //140 en adelante
            Image<Gray, byte> huefilter = imghue.InRange(new Gray(160), new Gray(255));
            //Filtro colores menos brillantes
            Image<Gray, byte> valfilter = imgval.InRange(new Gray(100), new Gray(255));
            //Filtro de saturación - quitar blancos 
            channels[1]._ThresholdBinary(new Gray(20), new Gray(255)); // Saturacion

            //Unir los filtros para obtener la imagen
            Image<Gray, byte> colordetimg = huefilter.And(valfilter).And(channels[1]);//aqui habia un Not()
            colordetimg._Erode(1);
            colordetimg._Dilate(1);
            //Colorear imagen
            //Image<Bgr, byte> ret = img;
            //var mat = img.Mat;
            //mat.SetTo(new MCvScalar(0, 0, 255), colordetimg);
            //mat.CopyTo(ret);
            //Image<Bgr, byte> imgout = ret.CopyBlank();//imagen sin fondo negro

            //ret._Or(imge);
            //Muestra imagen con los rojos destacados
            pictureBox2.Image = colordetimg.Bitmap;
            /*
             1. Edge detection (sobel)
             2. Dilation (10,1)
             3. FindContours
             4. Geometrical Constrints
             */
            //sobel
            Image<Gray, byte> sobel = colordetimg;
            //Image<Gray, byte> sobel = colordetimg.Convert<Gray, byte>().Sobel(0, 1, 5).AbsDiff(new Gray(0.0)).Convert<Gray, byte>().ThresholdBinary(new Gray(70), new Gray(255));
            //Image<Gray, byte> sobel2 = colordetimg.Convert<Gray, byte>().Sobel(1, 0, 5).AbsDiff(new Gray(0.0)).Convert<Gray, byte>().ThresholdBinary(new Gray(70), new Gray(255));
            //sobel.And(sobel2);
            //pictureBox3.Image = sobel.Bitmap;
            Mat SE = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(10, 2), new Point(-1, -1));
            sobel = sobel.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Close, SE, new Point(-1, -1), 15, Emgu.CV.CvEnum.BorderType.Reflect, new MCvScalar(255));
            Emgu.CV.Util.VectorOfVectorOfPoint contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            Mat m = new Mat();
            
            CvInvoke.FindContours(sobel, contours, m, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
           // pictureBox3.Image = sobel.Bitmap;
            
            List<Rectangle> list = new List<Rectangle>();

            for (int i = 0; i < contours.Size; i++)
            {
                Rectangle brect = CvInvoke.BoundingRectangle(contours[i]);

                double ar = brect.Width / brect.Height;
                if (ar > 2 && brect.Width > 25 && brect.Height > 8 && brect.Height < 100)
                {
                    list.Add(brect);
                }
            }
            //MessageBox.Show(list.ElementAt(0).ToString());

            Image<Bgr, byte> imgout = img.CopyBlank();
            foreach (var r in list)
            {
                CvInvoke.Rectangle(colordetimg, r, new MCvScalar(0, 0, 255), 2);
                CvInvoke.Rectangle(imgout, r, new MCvScalar(0, 255, 255), -1);
            }
            imgout._And(img);
            //pictureBox1.Image = img.Bitmap;
           pictureBox3.Image = imgout.Bitmap;

            //double fps = 15;
            //await Task.Delay(1000 / Convert.ToInt32(fps));

        }
    }
}
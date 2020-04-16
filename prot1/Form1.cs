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

        private void button1_Click(object sender, EventArgs e)
        {

        }
    }
}

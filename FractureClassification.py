import vtk, qt, ctk, slicer
import os
import numpy as np
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import logging

# Load PyTorch library
try:
  import torch
  import torch.nn as nn
  import torchvision
  from torchvision import transforms
  DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' # define current device
except:
  logging.error('PyTorch is not installed. Please, install PyTorch...')

# Load OpenCV
try:
  import cv2
except:
  logging.error('OpenCV is not installed. Please, install OpenCV...')


#------------------------------------------------------------------------------
#
# FractureClassification
#
#------------------------------------------------------------------------------
class FractureClassification(ScriptedLoadableModule):
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Fracture Classification"
    self.parent.categories = ["Ultrasound AI"]
    self.parent.dependencies = []
    self.parent.contributors = ["Nestor Manuel Acosta Rivero (ULPGC), David Garcia Mato (Ebatinca)"]
    self.parent.helpText = """ Module to classify and detect fractures in US images using deep learning. """
    self.parent.acknowledgementText = """EBATINCA, S.L."""


#------------------------------------------------------------------------------
#
# FractureClassificationWidget
#
#------------------------------------------------------------------------------
class FractureClassificationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

  def __init__(self, parent):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    #Flags
    self.showHeatMap = False
    self.showSegmentation = False

    # Create logic class
    self.logic = FractureClassificationLogic(self)

    # DEVELOPMENT
    slicer.FractureClassificationWidget = self

  #------------------------------------------------------------------------------
  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Set up UI
    self.setupUi()

    # Setup connections
    self.setupConnections()

    # The parameter node had defaults at creation, propagate them to the GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def cleanup(self):
    self.disconnect()

  #------------------------------------------------------------------------------
  def enter(self):
    """
    Runs whenever the module is reopened
    """

    # Update GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def exit(self):
    """
    Runs when exiting the module.
    """
    pass

  #------------------------------------------------------------------------------
  def setupUi(self):    
    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/FractureClassification.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Customize widgets
    self.ui.modelPathLineEdit.currentPath = self.logic.defaultModelFilePath

  #------------------------------------------------------------------------------
  def setupConnections(self):    
    self.ui.inputSelector.currentNodeChanged.connect(self.onInputSelectorChanged)
    self.ui.loadModelButton.connect('clicked(bool)', self.onLoadModelButton)
    self.ui.startClassificationButton.connect('clicked(bool)', self.onStartClassificationButton)
    self.ui.showHeatmapButton.connect('clicked(bool)', self.onShowHeatMap)
    self.ui.showSegmentationButton.connect('clicked(bool)', self.onShowSegmentation)

  #------------------------------------------------------------------------------
  def disconnect(self):
    self.ui.inputSelector.currentNodeChanged.disconnect()
    self.ui.loadModelButton.clicked.disconnect()
    self.ui.startClassificationButton.clicked.disconnect()
    self.ui.showHeatmapButton.clicked.disconnect()
    self.ui.showSegmentationButton.clicked.disconnect()

  #------------------------------------------------------------------------------
  def updateGUIFromMRML(self, caller=None, event=None):
    """
    Set selections and other settings on the GUI based on the parameter node.

    Calls the updateGUIFromMRML function of all tabs so that they can take care of their own GUI.
    """    
    # Display selected volume in red slice view
    inputVolume = self.ui.inputSelector.currentNode()
    if inputVolume:
      self.logic.displayVolumeInSliceView(inputVolume)

    # Activate buttons
    self.ui.startClassificationButton.enabled = (self.ui.inputSelector.currentNode() != None and self.logic.classificationModel != None)
    self.ui.showHeatmapButton.enabled = (self.logic.classificationFlag == True)
    self.ui.showSegmentationButton.enabled = (self.logic.classificationFlag == True)

  #------------------------------------------------------------------------------
  def onInputSelectorChanged(self):
    # Update GUI
    self.updateGUIFromMRML()
 
  #------------------------------------------------------------------------------
  def onLoadModelButton(self):
    # Acquire path from the line in UI
    modelFilePath = self.ui.modelPathLineEdit.currentPath

    # Load model using the function in the logic section
    self.logic.loadModel(modelFilePath)

    # Update GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def onStartClassificationButton(self):
    # Get input volume
    nodeName = self.ui.inputSelector.currentNode().GetName()
    
    # Classification
    [fracture_probability, heatMap] = self.logic.startClassification(nodeName)

    # Display fracture probability in UI
    self.ui.fractureProbabilityLabel.text = str(fracture_probability)    

    # Display heatmap
    self.logic.convertHeatMapToVolume(heatMap)
    
    # Segment heatmap and display segmentation
    self.logic.heatMapSegmentation(heatMap)

    # Update GUI
    self.updateGUIFromMRML()
  
  #------------------------------------------------------------------------------
  def onShowHeatMap(self):
    # Variable creation
    layoutManager = slicer.app.layoutManager()
    sliceCompositeNode = layoutManager.sliceWidget('Red').sliceLogic().GetSliceCompositeNode()

    # Update visibility
    if self.showHeatMap == False :
      self.showHeatMap = True # Flag change
      sliceCompositeNode.SetForegroundOpacity(0.5)
      self.ui.showHeatmapButton.text = 'Hide HeatMap'
    else:
      self.showHeatMap = False # Flag change
      sliceCompositeNode.SetForegroundOpacity(0)
      self.ui.showHeatmapButton.text = 'Show HeatMap'

    # Update GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def onShowSegmentation(self):
    # Variable creation
    segmentation = slicer.util.getNode('Segmentation')
    segmentId = segmentation.GetSegmentation().GetSegmentIdBySegmentName('Segmentation')

    # Update visibility
    if self.showSegmentation == False :
      self.showSegmentation = True # Flag change
      segmentation.GetDisplayNode().SetSegmentVisibility(segmentId, True)
      self.ui.showSegmentationButton.text = 'Hide Segmentation'
    else:
      self.showSegmentation = False # Flag change
      segmentation.GetDisplayNode().SetSegmentVisibility(segmentId, False)
      self.ui.showSegmentationButton.text = 'Show Segmentation'

    # Update GUI
    self.updateGUIFromMRML()


#------------------------------------------------------------------------------
#
# FractureClassificationLogic
#
#------------------------------------------------------------------------------
class FractureClassificationLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
  
  def __init__(self, widgetInstance, parent=None):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.moduleWidget = widgetInstance

    # Flag
    self.classificationFlag = False

    # Input image array and other properties
    self.inputImage = None
    self.numRows = None
    self.numCols = None

    # Classification model and HeatMap model
    self.defaultModelFilePath = self.moduleWidget.resourcePath('D:/UltrasoundAI/UltrasoundAI/FractureClassification/Resources/Model/BEST ResNet18/checkpoints/best_model_epoch=109.pth')
    self.classificationModel = None
    self.segmentationModel = None
    
    # Definition of transformations applied to image (transformation into tensor and normalizing values). These are necessary to go through the model.
    self.transformations = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(0.12967301527195704, 0.11952487479741893)
    ])
    # Red slice
    self.redSliceLogic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()

  #------------------------------------------------------------------------------
  def displayVolumeInSliceView(self, volumeNode):
    # Display input volume node in red slice background
    self.redSliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())
    self.redSliceLogic.FitSliceToAll()
  
  #------------------------------------------------------------------------------
  def loadModel(self, modelFilePath):
    """
    Loads PyTorch model for classification
    :param modelFilePath: path where the model file is saved
    :return: True on success, False on error
    """
    print('Loading model...')
    # Load classification model
    try:
      self.classificationModel = FractureModel() # Model instance
      checkpoint = torch.load(modelFilePath, map_location=torch.device(DEVICE)) # Checkpoint instance
      self.classificationModel.load_state_dict(checkpoint['state_dict'])  # Loads the weights stored in the 'state_dict' key
      self.classificationModel.to(DEVICE) 
      self.classificationModel.eval()
    except Exception as e:
      self.classificationModel = None
      logging.error("Failed to load classification model: {}".format(str(e)))
      return False
    
    # Load segmentation model
    try:
      self.segmentationModel = FractureModelMap() # HeatMap Model instance.
      checkpoint = torch.load(modelFilePath, map_location=torch.device(DEVICE)) # Checkpoint instance.
      self.segmentationModel.load_state_dict(checkpoint['state_dict'], strict = False) # Loads the weights stored in the 'state_dict' key.
      self.segmentationModel.to(DEVICE)
      self.segmentationModel.eval()
    except Exception as e:
      self.classificationModel = None
      logging.error("Failed to load segmentation model: {}".format(str(e)))
      return False
    
    return True
  
  #------------------------------------------------------------------------------
  def convertHeatMapToVolume(self, heatMap):
    """
    Generates heatmap visualization.
    :param heatMap: heatmap tensor.
    :return: True on success, False on error.
    """
    # Map transformations
    heatMap = np.array(heatMap)
    img_array = slicer.util.arrayFromVolume(self.inputImage)[0, :, :]

    # Data acquisition
    self.numRows = img_array.shape[0]
    self.numCols = img_array.shape[1]
    heatMap = cv2.resize(heatMap , (self.numCols, self.numRows)).astype(np.float16)
    
    # Remove existing node if any
    try:
      node = slicer.util.getNode('Heatmap')
      slicer.mrmlScene.RemoveNode(node)
    except:
      pass

    # Clone the node
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    itemIDToClone = shNode.GetItemByDataNode(self.inputImage)
    clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
    clonedNode = shNode.GetItemDataNode(clonedItemID)
    clonedNode.SetName('Heatmap')

    # Updating Red slice with the HeatMap
    slicer.util.updateVolumeFromArray(clonedNode, heatMap)
    sliceCompositeNode = self.redSliceLogic.GetSliceCompositeNode()
    sliceCompositeNode.SetForegroundVolumeID(clonedNode.GetID())
    sliceCompositeNode.SetForegroundOpacity(0)

    # Modify display properties
    heatmapVolumeDisplayNode = clonedNode.GetDisplayNode()
    heatmapVolumeDisplayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt') # lookup table
    heatmapVolumeDisplayNode.SetAutoThreshold(True)
    heatmapVolumeDisplayNode.SetAutoWindowLevel(True)

  #------------------------------------------------------------------------------
  def heatMapSegmentation(self, heatMap):
    """
    Generates heatmap segmentation and visualization.
    :param heatMap: heatmap tensor.
    :return: True on success, False on error.
    """
    # Map transformations
    heatMap = np.array(heatMap)
    img_array = slicer.util.arrayFromVolume(self.inputImage)[0, :, :]

    # Data acquisition
    self.numRows = img_array.shape[0]
    self.numCols = img_array.shape[1]
    heatMap = cv2.resize(heatMap , (self.numCols, self.numRows)).astype(np.float16)
    max = heatMap.max()

    # Threshold Segmentation
    segmentationMask = np.logical_and(heatMap[:,:] >= max*0.65, heatMap[:,:] <= max)

    # Remove existing segmentation node if any
    try:
      node = slicer.util.getNode('Segmentation')
      slicer.mrmlScene.RemoveNode(node)
    except:
      pass
    
    # Create new segmentation node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
    segmentationNode.CreateDefaultDisplayNodes()

    # Add segment to segmentation
    self.addSegmentFromNumpyArray(segmentationNode, segmentationMask, 'Segmentation', (1, 0, 0), False)

  #------------------------------------------------------------------------------
  def startClassification(self, nodeName):
    """
    US Image Classification 
    :param nodeName: Name of the current node.
    :return: percentage of prediction and heatmap.
    """
    # Node transformations
    self.inputImage = slicer.util.getNode(nodeName)
    img_array = slicer.util.arrayFromVolume(self.inputImage)[0, :, :]/255
    if nodeName == 'Image_Reference':
      img_array = np.rot90(np.rot90(img_array, k=1))
    img_array = cv2.resize(img_array, (224, 224)).astype(np.float16)
    image_trans = self.transformations(img_array).to(DEVICE).unsqueeze(0).float()

    # Proving that models exist 
    print('Starting classification...')
    if self.classificationModel is None or self.classificationModel is None:
     print("No model loaded")

    # Models inference    
    with torch.no_grad():  
      pred = torch.sigmoid(self.classificationModel(image_trans))
      feature_map = self.segmentationModel(image_trans)

    # Classification prediction Transformation for interpretation
    fractureProbability = round(pred.item()*100, 2)
    
    # Flag Updating
    self.classificationFlag = True

    # Predicted HeatMap transformations
    feature_map = feature_map.reshape((512, 49))
    weight_params = list(self.segmentationModel.model.fc.parameters())[0]
    weight = weight_params[0].detach()
    
    # Predicted HeatMap Computing
    heatMap = torch.matmul(weight, feature_map)
    heatMap = heatMap.reshape(7, 7).cpu()
    if nodeName == 'Image_Reference':
      heatMap = np.rot90(np.rot90(heatMap, k=1))
    # Output is the percentage of having a fracture located in the "percentage" instance and the computed heatmap
    return fractureProbability, heatMap

  #-------------------------------------------------------------------------------
  def addSegmentFromNumpyArray(self, outputSegmentation, input_np, segmentName, color, visibility):
    """
    Adds a new segment to segmentation node from a numpy array.
    :param outputSegmentation: segmentation node where the segment will be added
    :param input_np: input array containing image intensity values
    :param segmentName: name of the segment to be added
    :param labelValue: pixel value corresponding to region of interest
    :param inputVolume: reference volume node for segmentation
    :param color: color to be assigned to segment for display
    """
    # Creation of the segment
    emptySegment = slicer.vtkSegment()
    emptySegment.SetName(segmentName)
    emptySegment.SetColor(color)
    
    # Adding the segment to the segmentation node
    outputSegmentation.GetSegmentation().AddSegment(emptySegment)
    segmentId = outputSegmentation.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    outputSegmentation.GetDisplayNode().SetSegmentVisibility(segmentId, visibility)

    # Update the scene
    slicer.util.updateSegmentBinaryLabelmapFromArray(input_np, outputSegmentation, segmentId, self.inputImage)
   

#------------------------------------------------------------------------------
#
# FractureClassificationTest
#
#------------------------------------------------------------------------------
class FractureClassificationTest(ScriptedLoadableModuleTest):
  """This is the test case for your scripted module.
  """
  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    ScriptedLoadableModuleTest.runTest(self)


#------------------------------------------------------------------------------
#
# FractureModel
#
#------------------------------------------------------------------------------
class FractureModel(nn.Module):
    def __init__(self, weight=1):
      super().__init__()
        
      self.model = torchvision.models.resnet18()
      self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
      self.loss = torch.nn.BCEWithLogitsLoss(torch.tensor([weight]))
      
    def forward(self, data):
      pred = self.model(data)
      return pred


#------------------------------------------------------------------------------
#
# FractureModelMap
#
#------------------------------------------------------------------------------
class FractureModelMap(nn.Module):
    def __init__(self):
      super().__init__()
        
      self.model = torchvision.models.resnet18()
      self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
      self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])    
    
    def forward(self, data):
      feature_map = self.feature_map(data)
      return feature_map

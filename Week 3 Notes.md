# Week 3 - Feature Representation

She starts out with the explanation of the offsets from last class. This was a transform that allowed you achange a separator that did not go through the origing into one that did, in $d+1$ dimensions. She defines a transform from $R^d$ to $R^D$. It assumes that you have a linear separator in the lower dimensional space.

They provide an interesting example of this. Consider this: ![xor dataset](https://openlearninglibrary.mit.edu/assets/courseware/v1/7113a168fd1cb279b0a1548c7e16c08c/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_1-crop.png).

There's no linear separator for this in two dimensions. So let's take a low dimensional dataset and transform into a higher-dimensional one, and look for a separator there. **Note that a separator in 1-D is just a point!**:

![1-d ex](https://openlearninglibrary.mit.edu/assets/courseware/v1/083f20fe82de4d0b45c86e695e4d142c/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_2-crop.png)

Clearly, this has no separator. But let's try a transformation $\phi(x) = [x,x^2]$. This gives us: ![in the phi space](https://openlearninglibrary.mit.edu/assets/courseware/v1/bbc807b76c87fd03a606af741140257c/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_3-crop.png)
This clearly has a ton of separators! Note what happens here - the y value is the original x squared, so of course the +s would be positive and *higher* as they have larger magnitude.

But a linear separator in $\phi$ space is nonlinear in the original space. Here's a good example. Let's say we have a separator $x^2-1=0$ (note that this is what is depicted above - x^2 is the y axis, and anything above 1 is labeled as positive). This labels anything $x^2-1>0$ as positive. What does this correspond to in 1-D space? We need to ask "what x values have the property s.t. $x^2-1=0$?" Clearly, $+1,-1$. This defines our 1-D separator: ![1-d version of sep](https://openlearninglibrary.mit.edu/assets/courseware/v1/2df7785d97c451a36a9ff9b7683f1e2a/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_4-crop.png)

This is the basis of *kernel methods*, and is the motivation for multi-layer neural nets.

Obv, there are many ways to construct $\phi$. Some are systemic, domain independent (*polynomial basis*), others depend on the semantics (meaning) of the original features, so we design with the domain in mind.
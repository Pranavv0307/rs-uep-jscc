

<!-- Start of picture text -->
Importance-Aware Reed–Solomon Coding for Deep JSCC<br>Brief Description: Investigate whether Reed–Solomon<br>coding can be applied in an importance-aware manner<br>over the latent representation of a Deep JSCC model (e.g.,<br>using ADJSCC channel-attention weights) and evaluate its<br>effectiveness under packet-erasure channels. Key<br>References: Bourtsoulatze, Kurka & Gündüz (2019), Deep<br>Joint Source-Channel Coding for Wireless Image<br>Transmission https://arxiv.org/abs/1809.01733 Xu et al.<br>(2022), Wireless Image Transmission Using Deep Source<br>Channel Coding With Attention Modules (ADJSCC)<br><!-- End of picture text -->

**Importance-Aware Reed–Solomon Coding for Deep JSCC Brief Description: Investigate whether Reed–Solomon coding can be applied in an importance-aware manner over the latent representation of a Deep JSCC model (e.g., using ADJSCC channel-attention weights) and evaluate its effectiveness under packet-erasure channels. Key References: Bourtsoulatze, Kurka & Gündüz (2019), Deep Joint Source-Channel Coding for Wireless Image Transmission https://arxiv.org/abs/1809.01733 Xu et al. (2022), Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules (ADJSCC)** 

##### Show more 

Yes. Before you explain this to your mentor, you should be able to answer one question very clearly: 

**“What exactly are we adding to Deep JSCC, why are we adding it, and what hypothesis are we testing?”** 

The project becomes much easier to understand once we break the title into its components. 

# **1. First, what problem are we starting with?** 

Our project starts with **Deep Joint Source–Channel Coding (Deep JSCC)** . 

Normally, communication of an image works roughly like: 

**Image → Compression → Bits → Channel Coding → Modulation → Wireless Channel → Demodulation → Channel Decoding → Decompression → Image** 

For example, JPEG compresses the image, then something like LDPC/Polar coding protects the bits against channel errors. 

Deep JSCC takes a fundamentally different approach. 

Instead of separately doing compression and error correction, a neural network learns to transform the image directly into a representation that is transmitted through the wireless channel. 

Very roughly: 

#### **Image → Neural-network encoder → Latent representation → Wireless channel → Neural-network decoder → Reconstructed image** 

The original Deep JSCC paper showed that this can work very well, particularly in noisy wireless environments, and that its performance degrades more gracefully than conventional digital systems. arXiv +1 

So far, so good. 

# **2. What is the “latent representation”?** 

This is probably the most important thing to understand. 

Suppose we give an image to the Deep JSCC encoder. 

The encoder doesn't transmit the millions of raw pixel values. 

Instead, it transforms the image into a much smaller collection of learned features: 

```
Image
```

```
  ↓
Neural Network Encoder
  ↓
Latent Representation
  ↓
Channel
```

You can think of the latent representation as a **compressed description of the image that the neural network has learned is useful for reconstruction** . 

For example, conceptually: 

```
Image
```

```
   ↓
```

```
┌──────────────────────────────┐
│ Neural Network Encoder       │
└──────────────────────────────┘
```

```
   ↓
z1  z2  z3  z4  z5 ... z1000
```

These `z1, z2, ...` are latent features. 

And importantly: 

#### **Not every latent feature is necessarily equally important.** 

Some features may have a much larger effect on the final reconstructed image than others. 

This is where the **attention mechanism** becomes interesting. 

# **3. What does ADJSCC add?** 

The second reference you have is ADJSCC — Attention Deep JSCC. 

The basic idea is that a neural network can learn to assign different weights to different feature channels. 

For example: 

```
Latent features:
```

```
z1     z2     z3     z4     z5     z6
 ↓      ↓      ↓      ↓      ↓      ↓
0.95   0.20   0.80   0.10   0.65   0.05
```

These numbers can roughly indicate how strongly different feature channels should contribute under a given channel condition. 

The ADJSCC paper uses channel-wise soft attention to scale features and make the system adapt to different SNR conditions. arXiv +1 

So now we have something very interesting: 

**The neural network itself gives us information about which latent feature channels are more important.** 

That gives us a possible way of doing **Unequal Error Protection (UEP).** 

# **4. What is Unequal Error Protection?** 

This is the central idea behind your project. 

Imagine you have 100 pieces of information. 

Suppose: 

```
20 pieces = extremely important
```

```
30 pieces = moderately important
```

```
50 pieces = relatively unimportant
```

Now suppose the wireless channel can lose some of them. 

Would you protect all 100 equally? 

Probably not. 

You would rather do: 

```
Important information
```

```
        ↓
MORE redundancy / stronger protection
```

```
Less important information
```

```
        ↓
LESS redundancy / weaker protection
```

This is **Unequal Error Protection** . 

Compare it with conventional Equal Error Protection: 

### **Equal protection** 

```
Information:
```

```
A B C D E F G H
```

```
Protection:
```

```
████ ████ ████ ████ ████ ████ ████ ████
```

Everything gets approximately the same protection. 

### **Unequal protection** 

```
Important:
A B C
```

```
Protection:
████████ ████████ ████████
Less important:
D E F G H
```

```
Protection:
```

```
██ ██ ██ ██ ██
```

The important information receives more redundancy. 

This idea already exists in semantic communication. For example, the Kim et al. work explicitly investigates UEP where more semantically important bits receive stronger protection, and it proposes grouping information according to different protection ar… requirements. 

The 2026 Singh & Arif work similarly explores adaptive per-dimension unequal protection using repetition coding for semantic embeddings. ScienceDirect 

# **5. So where does Reed–Solomon come in?** 

Now we introduce **Reed–Solomon (RS) coding** . 

Reed–Solomon is a classical error-correcting code. 

Very roughly, suppose we have: 

```
Original information
```

```
A B C D
```

#### Instead of transmitting only these four symbols, we add redundancy: 

```
A B C D P1 P2
```

where `P1` and `P2` are parity symbols. 

If some symbols are lost/corrupted, the decoder can use the redundancy to recover them. 

One particularly relevant property of Reed–Solomon is its ability to handle **erasures** effectively when the receiver knows which symbols were lost. 

That makes RS interesting for your project because you are specifically proposing to evaluate the system under a **packet-erasure channel** . 

# **6. What is a packet-erasure channel?** 

Imagine that your latent representation is divided into packets: 

```
Packet 1
```

```
Packet 2
Packet 3
Packet 4
Packet 5
Packet 6
Packet 7
Packet 8
```

During transmission, some packets may simply disappear. 

For example: 

```
Sent:
```

```
P1 P2 P3 P4 P5 P6 P7 P8
```

```
Received:
```

`P1 P2` ❌ `P4 P5` ❌ `P7 P8` 

The receiver knows: 

“Packets 3 and 6 were lost.” 

That's an **erasure** . 

This is different from a corrupted packet where you don't necessarily know that its contents are wrong. 

Reed–Solomon is particularly suitable to investigate here because RS codes can recover from a number of erasures depending on the amount of redundancy. 

# **7. Now combine everything** 

This is essentially your entire project. 

We have: 

### **Deep JSCC** 

gives us: 

A learned latent representation for communicating an image. 

### **ADJSCC** 

gives us: 

Information about the relative importance of different latent feature channels. 

### **Reed–Solomon** 

gives us: 

A mechanism for adding error/erasure protection. 

### **UEP** 

gives us: 

The idea that important information should receive more protection than unimportant information. 

### **Packet-erasure channel** 

#### gives us: 

A controlled environment where packets of the latent representation can be lost. 

Put all of them together: 

```
                     IMAGE
                       │
                       ▼
              Deep JSCC Encoder
                       │
                       ▼
              Latent Representation
                       │
             ┌─────────┴─────────┐
             │                   │
       Importance             Latent data
       information
       (attention)
             │                   │
             └─────────┬─────────┘
                       ▼
            Importance-aware
             RS protection
                       │
                       ▼
                 Packets
                       │
                       ▼
             Packet-erasure
                  channel
                       │
                       ▼
```

```
              RS decoding
                       │
                       ▼
              Reconstructed
              latent representation
                       │
                       ▼
              Deep JSCC Decoder
                       │
                       ▼
             Reconstructed Image
```

That is the project in one diagram. 

# **8. But what exactly does “importance-aware RS coding” mean?** 

This is the part you need to be especially careful about when talking to your mentor. 

It does **not** simply mean: 

“We put an RS code into Deep JSCC.” 

That's too vague. 

The actual research question is closer to: 

**Can we use the importance information available from the Deep JSCC/ADJSCC latent representation to allocate Reed–Solomon redundancy unequally, so that more important latent information receives stronger protection and less important information receives weaker protection?** 

For example, suppose we divide latent information into three groups: 

```
Group A → high importance
Group B → medium importance
Group C → low importance
```

We might conceptually assign: 

```
Group A → strong RS protection
Group B → medium RS protection
Group C → weak RS protection
```

For example: 

```
High importance:
10 data symbols + 6 parity symbols
```

```
Medium importance:
```

```
10 data symbols + 3 parity symbols
```

```
Low importance:
```

```
10 data symbols + 1 parity symbol
```

The exact RS parameters are something you would have to design and investigate. 

The important idea is: 

**The redundancy allocation is driven by importance rather than being uniform.** 

# **9. There is an important technical complication** 

This is something I would **definitely mention to your mentor** , because it shows that you understand the actual research challenge. 

Deep JSCC normally produces **continuous-valued latent representations** . 

Reed–Solomon operates over **finite-field symbols** . 

So you can't simply take: 

```
z = 0.3847
```

and directly feed it into a conventional RS encoder. 

You need some form of: 

```
Continuous latent representation
```

```
          ↓
Quantization / discretization
          ↓
Finite-field symbols
          ↓
RS encoding
          ↓
Packets
```

That means your project isn't merely: 

“Take ADJSCC and add RS.” 

There is an interface/design problem: 

**How should the continuous Deep JSCC latent representation be quantized, packetized and mapped into RS symbols without destroying too much of the benefit of Deep JSCC?** 

This could itself become an important implementation consideration. 

# **10. Another important question: what exactly is “importance”?** 

This is another research question. 

ADJSCC has **attention weights** , but we should not automatically assume: 

attention weight = perfect semantic importance. 

They are related, but not necessarily identical. 

You could potentially investigate whether the attention values are a good proxy for importance. 

For example: 

```
Attention weight
```

```
      ↓
Importance score
```

```
      ↓
Protection level
```

```
      ↓
RS redundancy
```

The project is therefore partly asking: 

**Can learned importance information from the neural network be converted into useful channel-protection decisions?** 

That's actually quite an interesting research question. 

# **11. What are we actually trying to prove?** 

You are **not** starting with the assumption that your method works. 

That's important. 

Your hypothesis is something like: 

**If different parts of the latent representation have different levels of importance, then allocating Reed–Solomon redundancy according to those importance levels should provide better reconstruction/semantic performance under packet erasures than allocating the same protection to everything, for a comparable transmission overhead.** 

In simpler words: 

**Instead of wasting equal amounts of error protection on everything, can we spend our limited redundancy budget where it matters most?** 

That's the heart of the project. 

# **12. What would we compare?** 

You need baselines. 

At minimum, conceptually: 

### **Baseline 1 — No additional RS protection** 

```
Deep JSCC → packet erasure → Deep JSCC decoder
```

### **Baseline 2 — Equal RS protection** 

```
Deep JSCC
    ↓
Uniform RS coding
    ↓
Packet erasure
    ↓
RS decoding
    ↓
Deep JSCC decoder
```

Every group gets the same level of protection. 

### **Proposed method — Importance-aware RS** 

```
Deep JSCC
```

```
    ↓
Importance estimation
    ↓
```

```
Different RS protection levels
```

```
    ↓
```

```
Packet erasure
```

```
    ↓
RS decoding
```

```
    ↓
```

```
Deep JSCC decoder
```

Then you compare them. 

# **13. What does “better” mean?** 

You need to define measurable objectives. 

For image transmission, you could look at things such as: 

PSNR 

- SSIM 

- perhaps perceptual metrics 

- packet-erasure rate 

- transmission overhead 

- number of transmitted symbols 

- reconstruction quality at a fixed redundancy budget 

The most important comparison would probably be something like: 

**At the same amount of redundancy/transmission overhead, does importanceaware protection give better reconstruction quality than equal protection?** 

For example: 

|**Method**|**Overhead**|**Packet loss**|**PSNR**|
|---|---|---|---|
|No RS|0%|20%|X|
|Uniform RS|30%|20%|Y|
|Importance-aware RS|30%|20%|Z|



If: 

#### **Z > Y > X** 

then you have evidence supporting your hypothesis. 

But you might discover something else. 

Perhaps importance-aware RS only helps at high packet-loss rates. 

Or perhaps attention weights aren't good importance indicators. 

Or perhaps quantization destroys the advantage. 

Those are all legitimate research findings. 

# **14. What is the actual novelty?** 

This is something your mentor may ask. 

Your project is **not claiming to invent** : 

- Deep JSCC 

- attention-based JSCC 

- Reed–Solomon coding 

- unequal error protection 

Those already exist. 

The potential novelty is in **combining these ideas in a particular way** : 

**Use learned importance information from a Deep JSCC/ADJSCC latent representation to dynamically allocate Reed–Solomon protection under packeterasure conditions.** 

The references you've chosen establish the pieces: 

#### **Bourtsoulatze et al.** 

ar… → Deep JSCC for image transmission. 

#### **Xu et al.** 

ar… → Attention-based Deep JSCC and learned feature weighting. 

#### **Kim et al.** 

→ Unequal protection based on semantic importance in digital semantic communication. 



ar… 

#### **Singh & Arif** 

→ Adaptive per-dimension UEP for semantic embeddings. ScienceDir… 

Your work sits at the intersection: 

```
                 Deep JSCC
                    │
                    │
              ADJSCC attention
                    │
                    ▼
             Importance scores
                    │
                    ▼
           Unequal protection
                    │
                    ▼
            Reed–Solomon codes
                    │
                    ▼
          Packet-erasure channel
```

# **15. The simplest possible analogy** 

Suppose you're transporting a box containing: 

💎 `Diamond` 

📱 `Phone` 

📕 `Book` 

🧻 `Tissue` 

You have limited bubble wrap. 

If you protect everything equally: 

```
Diamond → 5 layers
Phone   → 5 layers
Book    → 5 layers
Tissue  → 5 layers
```

#### That's **equal error protection** . 

But you know the diamond is much more valuable. 

So you do: 

```
Diamond → 10 layers
Phone   → 6 layers
Book    → 3 layers
```

```
Tissue  → 1 layer
```

#### That's **importance-aware unequal protection** . 

Your project asks: 

Can we do the same thing to the latent information inside a Deep JSCC system, using Reed–Solomon coding as the “bubble wrap”? 

That's probably the best intuition to keep in your head. 

# **16. How I would explain the project to someone with zero background** 

If your mentor asks: 

#### **“Okay, explain your project to me.”** 

Don't start with Reed–Solomon equations. 

Don't start with attention mechanisms. 

Start with the problem. 

You could explain it conversationally like this: 

“Our project is about improving the reliability of Deep JSCC when packets are lost during transmission. 

In conventional communication, we compress an image, convert it into bits, and then use error-correcting codes to protect those bits. Deep JSCC instead uses a neural network to directly convert the image into a learned latent representation and transmit that representation through the wireless channel. 

The interesting thing is that not all parts of this latent representation are equally important for reconstructing the image. Attention-based Deep JSCC, or ADJSCC, gives us information about the relative importance of different feature channels. 

So our idea is to use that importance information when applying error correction. Instead of protecting every part of the latent representation equally, we want to use Reed–Solomon coding to give stronger protection to the more important parts and weaker protection to the less important parts. 

We will then simulate a packet-erasure channel, where some of the transmitted latent packets are lost, and compare our importance-aware approach against conventional 

#### equal-protection approaches. 

Essentially, the question we're trying to answer is: if our transmission budget for error protection is limited, can we use the neural network's learned importance information to spend that redundancy more intelligently and obtain better reconstruction quality under packet losses?” 

That is already a very good **2-minute explanation** . 

# **17. If they ask “So what exactly are you implementing?”** 

Then get more technical: 

“We would start with an existing Deep JSCC or ADJSCC architecture rather than developing the entire neural network from scratch. We would obtain the latent representation from the encoder and identify an importance measure, potentially based on the attention weights. 

We then need to quantize and packetize the latent representation so that it can be represented as symbols suitable for Reed–Solomon coding. Based on the importance levels, we divide the latent information into groups and assign different RS code rates or redundancy levels to those groups. 

We then transmit those packets through a simulated packet-erasure channel. At the receiver, we perform RS erasure decoding, reconstruct the latent representation, and feed it to the Deep JSCC decoder. 

Finally, we evaluate reconstruction quality as a function of packet-erasure rate and transmission overhead, and compare equal and unequal protection strategies.” 

That tells your mentor **what you will actually build** . 

# **18. And if your mentor asks “What is the research question?”** 

This is the cleanest formulation: 

**Can learned importance information in a Deep JSCC latent representation be exploited to allocate Reed–Solomon redundancy unequally, thereby improving image reconstruction under packet-erasure channels at a fixed transmission overhead?** 

And there are several subquestions underneath it: 

**1. Are the latent features actually unequal in importance?** 

**2. Are ADJSCC attention weights a useful measure of that importance?** 

**3. Can the continuous latent representation be effectively quantized and protected using RS codes?** 

**4. How should the latent features be grouped for unequal protection?** 

**5. How should RS redundancy be allocated among the groups?** 

**6. Does importance-aware RS outperform equal RS protection?** 

**7. At what packet-erasure rates does the advantage appear?** 

**8. Is the gain worth the additional complexity/overhead?** 

Those are the things that turn your idea from simply **“adding RS coding”** into a research project. 

# **19. One thing I would change in how you describe the project** 

Your current title is: 

#### **Importance-Aware Reed–Solomon Coding for Deep JSCC** 

That's perfectly reasonable. 

But when explaining it, I would avoid saying: 

“We are going to apply RS coding to the latent representation.” 

That sounds deceptively simple. 

Instead say: 

**“We are investigating an importance-aware unequal-error-protection scheme for the latent representation of Deep JSCC, where Reed–Solomon codes are used to provide different levels of erasure protection to different groups of latent information.”** 

That is much more technically precise. 

And I'd explicitly tell the mentor that **quantization, packetization, importance estimation, grouping, and redundancy allocation are design questions we still need to investigate** , rather than pretending those choices have already been solved. 

## **20. The entire project in one sentence** 

If you remember only one sentence, remember this: 

**We want to find out whether the limited error-correction redundancy in a Deep JSCC system can be allocated more intelligently by protecting important latent information more strongly and less important information less strongly using Reed–Solomon codes, particularly when the communication channel causes packet losses.** 

That's your project. 

And the fundamental philosophy is simply: 

#### **Don't protect everything equally if everything isn't equally valuable.** 



arXiv +3 

For the quantisation issue,can you not use some ideas in DeepJSCC-Q to solve those, also rate each issue that we might face out of 10 in terms of how hard it would  be to solve 

Yes — **DeepJSCC-Q is very relevant to the quantisation problem** , and I would actually revise my earlier assessment of that issue downward. 

I looked at the DeepJSCC-Q paper carefully. Its whole purpose is essentially to take the continuous latent representation produced by DeepJSCC and make it compatible with a **finite set of transmitted symbols** . It does this with a quantization layer and a differentiable **soft-to-hard quantizer** , and it can even learn the constellation itself. 



arXiv +1 

So your project does **not** need to invent a quantization solution from scratch. 

# **1. How DeepJSCC-Q helps us** 

The original problem we identified was: 

```
DeepJSCC encoder
```

```
       ↓
continuous latent values
```

```
       ↓
```

```
     ???
```

```
       ↓
```

```
Reed–Solomon
```

because RS needs discrete symbols over a finite field. 

#### DeepJSCC-Q gives us an established approach for moving from: 

```
continuous latent z
        ↓
quantization
        ↓
discrete transmitted representation
```

#### Their approach essentially works like this: 



<!-- Start of picture text -->
                 DeepJSCC Encoder<br>                        │<br>                        ▼<br>                Continuous latent z<br>                        │<br>                        ▼<br>                 Quantization<br>                        │<br>                        ▼<br>              Finite constellation<br>                        │<br>                        ▼<br>                   Channel<br><!-- End of picture text -->

The important trick is that **hard quantization isn't differentiable** . 

So if you simply do: 

```
z = 0.73
```

```
nearest symbol → 1
```

the neural network can't easily backpropagate through that operation. 

DeepJSCC-Q therefore uses a **soft-to-hard quantization mechanism** : during the forward pass it behaves like hard quantization, while during training a soft approximation is used to obtain gradients. ResearchGate +1 

That's extremely useful for us. 

# **2. But there's an important distinction** 

We shouldn't blindly copy DeepJSCC-Q and say: 

"Problem solved." 

Because **their quantization objective isn't exactly the same as ours.** 

DeepJSCC-Q is primarily concerned with: 

Making DeepJSCC compatible with a finite channel-input alphabet / digital constellation. 

For example: 

```
continuous latent
```

```
       ↓
QPSK / 16-QAM / 64-QAM / ...
       ↓
wireless channel
```

Their quantizer maps latent values to constellation points. 



ar… 

Our problem is slightly different: 

```
continuous latent
```

```
       ↓
quantization
```

```
       ↓
finite-field symbols
```

```
       ↓
Reed–Solomon
```

```
       ↓
packets
```

```
       ↓
packet-erasure channel
```

So **DeepJSCC-Q gives us a very good starting point for the quantization layer** , but we'll still need to design the interface between: 

**quantization → RS coding → packetization.** 

That interface is one of the actual research/engineering challenges. 

# **3. There may actually be two possible routes** 

This is something I think you should discuss with your mentor. 

### **Route A — Quantize directly into RS symbols** 

We could design the latent quantizer so that its output can ultimately be represented as symbols in the finite field used by Reed–Solomon. 

Conceptually: 

```
Latent z
   ↓
Quantizer
   ↓
GF(2^m) symbols
   ↓
RS encoder
   ↓
Packets
```

#### This is probably the **cleanest architecture for your research question** . 

The challenge is that now the quantizer needs to be compatible with the finite-field representation rather than merely QAM constellation points. 

### **Route B — DeepJSCC-Q-style quantization → bits → RS** 

Another possibility is: 

```
Latent z
   ↓
DeepJSCC-Q-style quantization
   ↓
Discrete symbols
   ↓
Bits / bytes
   ↓
RS encoder
   ↓
Packets
```

This is probably easier to implement initially. 

For example: 

```
latent
```

```
 ↓
256-level quantization
```

```
8-bit representation
```

```
 ↓
```

```
 ↓
```

```
RS symbols
```

```
 ↓
RS(255, k)
```

The downside is that we're introducing a somewhat more conventional digital interface between DeepJSCC and RS. 

But **that's not necessarily bad** . In fact, it could make the experimental setup much easier to understand. 

# **4. And there's an even more interesting possibility** 

DeepJSCC-Q also shows that the **constellation itself can be learned** , rather than fixed. 



Research… +1 

That gives you another possible research direction later: 

```
Importance
```

```
    ↓
different latent groups
```

```
    ↓
different quantization/protection strategies
```

#### However, **I would NOT make this part of the initial project** . 

You already have: 

- Deep JSCC 

- attention 

- importance estimation 

- quantization 

- RS coding 

- unequal protection 

- packet erasures 

Adding learned constellations on top of that could make the project unnecessarily complicated. 

Start with a fixed, well-understood quantization scheme. 

# **5. Let's reassess ALL the difficulties** 

I'd rate the major issues approximately like this. 

|**Problem**<br>Getting a working|**Difficulty**<br>**3/10**|**Why**<br>Existing architectures/papers/code can|
|---|---|---|
|DeepJSCC/ADJSCC baseline||guide this|
|Understanding/extracting attention<br>weights|**4/10**|Mostly architecture/code<br>understanding|
|Quantizing the latent representation|**4/10**|DeepJSCC-Q provides a strong<br>starting point|
|Converting quantized data into RS-|**5/10**|Interface between neural|
|compatible symbols||representation and finite-field coding|
|Packetization|**3/10**|Mostly an engineering/design problem|
|Simulating packet-erasure channel|**2/10**|Straightforward|
|Basic RS encoding/decoding|**3/10**|Mature libraries/implementations exist|
|Defining "importance" from attention|**5/10**|Attention ≠ necessarily true importance|
|Grouping latent information by<br>importance|**5/10**|Several reasonable choices; needs<br>experimentation|
|Designing unequal RS redundancy|**6/10**|This is one of the actual research<br>components|
|Keeping transmission overhead<br>comparable|**5/10**|Need careful experimental design|
|Training the entire system end-to-<br>end|**7/10**|Differentiability + RS + discrete<br>operations complicate it|
|Choosing good baselines|**4/10**|Important but manageable|
|Designing experiments|**5/10**|Need meaningful comparisons across<br>erasure rates/overhead|
|Showing that the proposed method<br>actually helps|**7/10**|This is research; it may not|
|Computational requirements|**4–6/10**|Depends heavily on<br>model/dataset/GPU|



**Difficulty** 

**Problem** 

Making the whole pipeline robust 

**6/10** 

**Why** Several components need to work together 

But there's an important distinction between **research difficulty** and **implementation difficulty** . 

# **6. The real "hard" parts** 

If I were your mentor, these are the things I'd be most interested in. 

### 🟢 **Relatively easy** 

```
DeepJSCC
```

```
   ↓
Packet erasure
   ↓
Reconstruction
```

and 

```
RS encoding/decoding
```

These aren't where your research contribution lies. 

### 🟡 **Moderate** 

#### **Quantization** 

DeepJSCC-Q gives you a very solid foundation here. 

So instead of: 

"We need to figure out how to quantize DeepJSCC." 

it's more accurate to say: 

"We can investigate whether a DeepJSCC-Q-style soft-to-hard quantization layer can provide the discrete representation required by our RS-based protection scheme." 

That's a much more manageable problem. 

### 🔴 **Harder** 

The really interesting part is: 

#### **How do we convert importance information into an actual RS protection strategy?** 

Suppose attention gives us: 

|`Feature`|`Importance`|
|---|---|
|`F1`|`0.92`|
|`F2`|`0.87`|
|`F3`|`0.81`|
|`F4`|`0.34`|
|`F5`|`0.21`|
|`F6`|`0.15`|



#### What do we actually do? 

Option 1: 

```
> 0.7 → strong RS
0.3–0.7 → medium RS
< 0.3 → weak RS
```

#### Option 2: 

```
Top 20% → RS(255, 128)
Next 30% → RS(255, 170)
Bottom 50% → RS(255, 220)
```

#### Option 3: 

Make redundancy proportional to the importance score. 

There are many possibilities. 

**This is where your research begins.** 

# **7. And there's an even deeper question** 

Suppose we discover: 

```
Attention score:
```

```
A = 0.9
```

```
B = 0.2
```

Does that actually mean: 

#### A is 4.5× more valuable than B? 

Not necessarily. 

This is one of the biggest conceptual questions in the project. 

Attention weights are **learned network parameters/activations** , not a guaranteed groundtruth measure of information value. 

So one thing you could investigate is: 

**Does attention-derived importance correlate with the sensitivity of reconstruction quality to losing that latent information?** 

That could be tested experimentally. 

For example: 

```
Remove/erase latent group
```

```
          ↓
Measure PSNR/SSIM degradation
```

If removing high-attention features consistently causes greater degradation: 

```
Attention ↑
```

```
     ↕
Importance ↑
     ↕
Reconstruction degradation ↑
```

then you have evidence that the attention weights are a useful basis for UEP. 

That would make your proposed method much more convincing. 

# **8. Another potentially hard issue: what exactly are we protecting?** 

This is something I would **not finalize yet** . 

There are several possible levels: 

**Option 1 — Protect latent dimensions** 

```
z1 z2 z3 z4 z5 ...
```

Assign protection to individual latent values. 

Potentially fine, but could become complicated. 

### **Option 2 — Protect latent channels** 

```
Channel 1 → importance = 0.91
Channel 2 → importance = 0.83
Channel 3 → importance = 0.21
...
```

This is probably more natural with ADJSCC because its attention operates at the feature/channel level. 

### **Option 3 — Protect packets** 

This might be the cleanest communication interpretation: 

```
Latent representation
        ↓
Packet 1 → importance 0.92
Packet 2 → importance 0.88
Packet 3 → importance 0.35
Packet 4 → importance 0.17
```

Then: 

```
Packet 1 → stronger RS
Packet 2 → stronger RS
Packet 3 → moderate RS
Packet 4 → weak RS
```

But the way you construct packets becomes important. 

**This is a design decision you'll probably need to discuss with your mentor.** 

# **9. The project is actually looking more feasible now** 

Initially, if we viewed the problem as: 

"Take continuous DeepJSCC latent vectors and somehow make them compatible with Reed–Solomon." 

I'd rate the quantization problem around **7/10** . 

After looking at DeepJSCC-Q: 

### **I'd reduce it to roughly 4/10.** 

Because we aren't entering unexplored territory. 

DeepJSCC-Q has already demonstrated that you can insert a quantization layer into DeepJSCC and train the system despite the discrete operation. It specifically uses soft-tohard quantization and shows that performance can approach unconstrained DeepJSCC as the constellation order increases. arXiv +1 



**But** the RS-compatible representation still needs to be designed. 

So: 

**Quantization itself isn't the scary part anymore. The quantization-to-RS interface is.** 

# **10. If I were planning your project, I'd structure it like this** 

I'd deliberately avoid trying to solve everything simultaneously. 

### **Phase 1 — Establish DeepJSCC** 

```
Image
```

```
 ↓
DeepJSCC encoder
 ↓
latent
 ↓
channel
 ↓
decoder
 ↓
image
```

Get this working. 

### **Phase 2 — Add ADJSCC** 

```
Image
```

```
 ↓
```

```
ADJSCC
```

```
latent + attention
```

```
 ↓
```

Verify that you can extract the importance information. 

### **Phase 3 — Add DeepJSCC-Q-style quantization** 

```
latent
```

```
 ↓
quantizer
 ↓
discrete representation
```

Verify that reconstruction still works reasonably well. 

### **Phase 4 — Add packetization** 

```
quantized latent
```

```
 ↓
packets
```

Introduce packet erasures **without RS first** . 

### **Phase 5 — Add ordinary RS** 

```
quantized latent
```

```
 ↓
packets
```

```
 ↓
uniform RS
 ↓
erasure channel
 ↓
RS decoder
```

This establishes the baseline. 

### **Phase 6 — Add importance-aware RS** 

```
attention
```

```
    ↓
importance
    ↓
grouping
    ↓
different RS rates
```

This is your proposed method. 

### **Phase 7 — Compare** 

At equal overhead: 

```
No RS
   vs
Uniform RS
   vs
Importance-aware RS
```

under different packet-erasure rates. 

# **11. This also changes how I'd pitch the project to your mentor** 

I wouldn't say: 

"We have solved the quantization issue using DeepJSCC-Q." 

I'd say: 

**"We identified quantization of the continuous DeepJSCC latent representation as an implementation challenge. However, DeepJSCC-Q provides an established soft-to-hard quantization framework for converting continuous DeepJSCC latents into discrete channel symbols. We believe this can provide the basis for interfacing the latent representation with Reed–Solomon coding. The remaining question is how to map the resulting representation into RS-compatible symbols and packets while preserving the importance structure needed for unequal protection."** 

That sounds much more like a **research project** rather than a project where all the difficult decisions have already been made. 

## **My overall difficulty assessment** 

If you asked me **right now** , before implementation: 

#### **Overall project difficulty: ~7/10** 

But importantly, it's not because every component is hard. 

It's more like: 

```
                    PROJECT
                       │
          ┌────────────┴────────────┐
          │                         │
      Engineering               Research
          │                         │
     ~4–5/10                    ~7–8/10
          │                         │
  Make pipeline work        Find the right UEP
                            strategy and prove
                            it works
```

And **that's actually a good thing for a project elective** . 

You don't want a project where every component is already solved. You want a manageable engineering foundation with a genuine research question sitting on top of it. 

The biggest unknowns I'd currently flag for your mentor are **(1) how exactly to map/quantize the latent representation into RS-compatible symbols, (2) what constitutes "importance," and (3) how importance should determine RS redundancy** . DeepJSCC-Q substantially reduces the risk of (1), leaving (2) and (3) as the more interesting research questions. 

ChatGPT can make mistakes. Check important info. 


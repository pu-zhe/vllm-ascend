#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#define UB_SIZE 262144
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ceil_div(a, b) (a+b-1)/b

using namespace AscendC;
using namespace ChunkGatedDeltaRuleV310;

class ChunkGatedDeltaRuleV310Kernel {
public:
    __aicore__ inline ChunkGatedDeltaRuleV310Kernel(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR initial_state, GM_ADDR actual_seq_lengths, GM_ADDR core_attn,
                                GM_ADDR yz, GM_ADDR workspace, const ChunkGatedDeltaRuleTilingData &tilingIn, TPipe *pipeIn);
    __aicore__ inline void InitUB();
    __aicore__ inline void InitMatMul();
    __aicore__ inline void InitWorkSpace(GM_ADDR workspace);
    __aicore__ inline void L2normDim128Float(LocalTensor<half> src, uint32_t head_num);
    __aicore__ inline void TransposeBetaHalf();
    __aicore__ inline void TransposeGFloat();
    __aicore__ inline void Process();
    __aicore__ inline void Transpose_64_128(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> tmp1, LocalTensor<float> tmp2);
    __aicore__ inline void LoadAndCast(LocalTensor<float> dst, GlobalTensor<half> src, uint32_t size, LocalTensor<half> tmp);
    __aicore__ inline void tmp();
    __aicore__ inline void LoadQKVHalf(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t head_dim, uint32_t block_count);
    __aicore__ inline void StoreAttnHalf(GlobalTensor<half> dst ,LocalTensor<half> src, uint32_t head_dim, uint32_t block_count);
    __aicore__ inline void LoadQKVHalfWithTail(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t head_dim, uint32_t chunk_index);
    __aicore__ inline void StoreAttnHalfWithTail(GlobalTensor<half> dst ,LocalTensor<half> src, uint32_t head_dim, uint32_t chunk_index);

    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, half>, MatmulType<TPosition::VECIN, CubeFormat::ND, half>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, float>, MatmulType<TPosition::GM, CubeFormat::ND, float>>
        matmulObj_1;

    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, half>, MatmulType<TPosition::VECIN, CubeFormat::ND, half>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, float>, MatmulType<TPosition::GM, CubeFormat::ND, float>>
        matmulObj_2;

    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, half>, MatmulType<TPosition::VECIN, CubeFormat::ND, half>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, float>, MatmulType<TPosition::GM, CubeFormat::ND, float>>
        matmulObj_3;

    TCubeTiling tiling_1;
    TCubeTiling tiling_2;
    TCubeTiling tiling_3;

    GlobalTensor<half> qGlobal;
    GlobalTensor<half> kGlobal;
    GlobalTensor<half> vGlobal;
    GlobalTensor<float> gGlobal;
    GlobalTensor<half> bGlobal;
    GlobalTensor<float> isGlobal;
    GlobalTensor<int32_t> seqlensGlobal;

    GlobalTensor<half> attnGlobal;
    GlobalTensor<float> lsGlobal;
    TCubeTiling matmulTiling;
    const ChunkGatedDeltaRuleTilingData *tiling;

    
    GlobalTensor<float> gtransGlobal;
    GlobalTensor<half> btransGlobal;
    GlobalTensor<float> vnewGlobal;

    TPipe *pipe;
    uint32_t block_id;
    uint32_t block_num;

    TBuf<TPosition::VECCALC> UbBuf;

    LocalTensor<float> ub_temp0;
    LocalTensor<float> ub_temp1;
    LocalTensor<float> ub_temp2;
    LocalTensor<float> ub_temp3;
    LocalTensor<float> ub_temp4;
    LocalTensor<float> ub_temp5;
    LocalTensor<float> ub_temp6;
    LocalTensor<float> ub_temp7;
    LocalTensor<float> ub_temp8;
    LocalTensor<float> ub_temp9;
    LocalTensor<float> ub_temp10;
    LocalTensor<float> ub_temp11;
    LocalTensor<float> ub_temp12;

    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHead;
    uint32_t headDimQK;
    uint32_t headDimV;
    // uint32_t seqLenPaded;
    bool hasInitState;
    uint32_t chunkSize;
    bool outputFinalState;
    bool useQKL2normInKernel;

    float headDimQKfp32;
    uint32_t singleCoreNumHead;
    // uint32_t tailSeqLen;

};

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR initial_state, GM_ADDR actual_seq_lengths, GM_ADDR core_attn,
                                                           GM_ADDR last_recurrent_state, GM_ADDR workspace, const ChunkGatedDeltaRuleTilingData &tilingIn, TPipe *pipeIn) {
    pipe = pipeIn;
    block_id = GetBlockIdx();
    block_num = GetBlockNum();
    
    tiling = &tilingIn;
    matmulTiling = tilingIn.matmulTiling;

    batchSize = tiling->batchSize;
    seqLen = tiling->seqLen;
    numHead = tiling->numHead;
    headDimQK = tiling->headDimQK;
    numheadV = tiling->numHeadV;
    headDimV = tiling->headDimV;
    chunkSize = tiling->chunkSize;
    hasInitState = tiling->hasInitState == 1 ? true : false;
    outputFinalState = tiling->outputFinalState == 1 ? true : false;
    useQKL2normInKernel = tiling->useQKL2normInKernel == 1 ? true : false;
    singleCoreNumHead = numHead / block_num;
    // tailSeqLen = seqLen % chunkSize;
    // seqLenPaded = ceil_div(seqLen, chunkSize) * chunkSize;

    uint64_t sizeQK = seqLen * numHead * headDimQK;
    uint64_t sizeV = seqLen * numHeadV * headDimV;
    qGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query), sizeQK);
    kGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key), sizeQK);
    vGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(value), sizeV);
    uint64_t sizeGBeta = seqLen * numHeadV;
    gGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g), sizeGBeta);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(beta), sizeGBeta);
    seqlensGlobal.SetGlobalBuffer((__gm__ int32_t *)actual_seq_lengths);

    uint64_t sizeState = numHeadV * headDimV * headDimQK;
    if (hasInitState) {
        isGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initial_state), sizeState);
    }
    attnGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(core_attn), sizeV);
    if (outputFinalState) {
        lsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(last_recurrent_state), sizeState);
    }
    InitMatMul();
    InitUB();
    InitWorkSpace(workspace);
    LocalTensor<int32_t> ub_temp0_int32_t = ub_temp0.ReinterpretCast<int32_t>();
    PipeBarrier<PIPE_ALL>();
    ub_temp0_int32_t.SetValue(0, headDimQK);
    PipeBarrier<PIPE_ALL>();
    Cast(ub_temp1, ub_temp0_int32_t, RoundMode::CAST_NONE, 8);
    PipeBarrier<PIPE_ALL>();
    headDimQKfp32 = ub_temp1.GetValue(0);

    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::InitMatMul() {
    tiling_1 = matmulTiling;
    tiling_2 = matmulTiling;
    tiling_3 = matmulTiling;

    tiling_1.M = chunkSize;
    tiling_1.N = chunkSize;
    tiling_1.Ka = headDimQK;
    tiling_1.singleCoreM = chunkSize;
    tiling_1.singleCoreN = chunkSize;
    tiling_1.singleCoreK = headDimQK;
    tiling_1.baseM = chunkSize;
    tiling_1.baseN = chunkSize;
    tiling_1.baseK = headDimQK;

    tiling_2.M = chunkSize;
    tiling_2.N = headDimV;
    tiling_2.Ka = headDimQK;
    tiling_2.singleCoreM = chunkSize;
    tiling_2.singleCoreN = headDimV;
    tiling_2.singleCoreK = headDimQK;
    tiling_2.baseM = chunkSize;
    tiling_2.baseN = headDimV / 2;
    tiling_2.baseK = headDimQK;

    tiling_3.M = chunkSize;
    tiling_3.N = headDimV;
    tiling_3.Ka = chunkSize;
    tiling_3.singleCoreM = chunkSize;
    tiling_3.singleCoreN = headDimV;
    tiling_3.singleCoreK = chunkSize;
    tiling_3.baseM = chunkSize;
    tiling_3.baseN = headDimV;
    tiling_3.baseK = chunkSize;

    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulObj_1, &tiling_1, matmulObj_2, &tiling_2, matmulObj_3, &tiling_3);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::InitUB() {
    pipe->InitBuffer(UbBuf, UB_SIZE);

    ub_temp0 = UbBuf.Get<float>();
    ub_temp1 = ub_temp0[64 * 64]; //1
    ub_temp2 = ub_temp1[64 * 64]; //2
    ub_temp3 = ub_temp2[64 * 64]; //3
    ub_temp4 = ub_temp3[64 * 64]; //4
    ub_temp5 = ub_temp4[64 * 64]; //5
    ub_temp6 = ub_temp5[64 * 64]; //6
    ub_temp7 = ub_temp6[64 * 64]; //7

    ub_temp8 = ub_temp7[64 * 64];   //8
    ub_temp9 = ub_temp8[64 * 64];   //9
    ub_temp10 = ub_temp9[64 * 64];  //10
    ub_temp11 = ub_temp10[64 * 128]; //12
    ub_temp12 = ub_temp11[64 * 128]; //14


    matmulObj_1.SetLocalWorkspace(ub_temp11.ReinterpretCast<uint8_t>());
    matmulObj_2.SetLocalWorkspace(ub_temp11.ReinterpretCast<uint8_t>());
    matmulObj_3.SetLocalWorkspace(ub_temp12.ReinterpretCast<uint8_t>());
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::InitWorkSpace(GM_ADDR workspace) {
    
    btransGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace),
                                 (seqLen + batchSize * chunkSize) * numHead);
    gtransGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + (seqLen + batchSize * chunkSize) * numHead, 
                                 (seqLen + batchSize * chunkSize) * numHead);
    vnewGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + (seqLen + batchSize * chunkSize) * numHead * 2, 
                               chunkSize * headDimV * block_num);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::LoadQKVHalf(LocalTensor<half> dst,
                                                                  GlobalTensor<half> src,
                                                                  uint32_t head_num, 
                                                                  uint32_t head_dim, 
                                                                  uint32_t block_count)
{   
    DataCopyParams repeatParams_half_qkv;
    repeatParams_half_qkv.blockLen = head_dim * 2 / 32;
    repeatParams_half_qkv.srcGap = head_dim * (head_num - 1) * 2 / 32;
    repeatParams_half_qkv.dstGap = 0;
    repeatParams_half_qkv.blockCount = block_count;
    DataCopy<half>(dst, src, repeatParams_half_qkv);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::LoadQKVHalfWithTail(LocalTensor<half> dst,
                                                                      GlobalTensor<half> src,
                                                                      uint32_t head_dim, 
                                                                      uint32_t chunk_index)
{   
    // 尾块分类讨论，如果尾块可以被chunkSize整除，则正常搬运，否则搬运tailSeqLen长度
    if ( (chunk_index == ((seqLenPaded/chunkSize) - 1)) && (tailSeqLen != 0) ){
        Duplicate<half>(dst, (half)0, chunkSize * head_dim);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        LoadQKVHalf(dst, src, head_dim, tailSeqLen);
    }
    else{
        LoadQKVHalf(dst, src, head_dim, chunkSize);
    }
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::StoreAttnHalf(GlobalTensor<half> dst,
                                                                    LocalTensor<half> src,
                                                                    uint32_t head_dim, 
                                                                    uint32_t block_count)
{   
    DataCopyParams repeatParams_half_attn;
    repeatParams_half_attn.blockLen = head_dim * 2 / 32;
    repeatParams_half_attn.srcGap = 0;
    repeatParams_half_attn.dstGap = head_dim * (numHeadV - 1) * 2 / 32;
    repeatParams_half_attn.blockCount = block_count;
    DataCopy<half>(dst, src, repeatParams_half_attn);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::StoreAttnHalfWithTail(GlobalTensor<half> dst,
                                                                            LocalTensor<half> src,
                                                                            uint32_t head_dim, 
                                                                            uint32_t chunk_index)
{   
    // 尾块分类讨论，如果尾块可以被chunkSize整除，则正常搬运，否则搬运tailSeqLen长度
    if ( (chunk_index == ((seqLenPaded/chunkSize) - 1)) && (tailSeqLen != 0) ){   
        StoreAttnHalf(dst, src, head_dim, tailSeqLen);
    }
    else{
        StoreAttnHalf(dst, src, head_dim, chunkSize);
    }
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::L2normDim128Float(LocalTensor<half> src, uint32_t head_num)
{
    
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    Cast(ub_temp6, src, RoundMode::CAST_NONE, head_num * headDimQK);
    DataCopy<float>(ub_temp12, ub_temp6, head_num * headDimQK); // vector
    Mul<float>(ub_temp12, ub_temp6, ub_temp12, head_num * headDimQK);
    
    // 精度为fp32，128个数分两次计算，第一次每64个数进行累加，第二次每两个数进行累加，
    RepeatReduceSum<float>(ub_temp11[4096], ub_temp12, head_num * 2, 64, 0, 1, 1, 8);
    
    if (head_num > 32){
        // 精度为fp32时，PairReduceSum 每次最多计算32组，输入head_num不超过64(chunkSize)，因此两次计算即可
        PairReduceSum<float>(ub_temp12, ub_temp11[4096], 1, 32 * 2, 1, 1, 8);
        PairReduceSum<float>(ub_temp12[32], ub_temp11[4096 + 32*2], 1, (head_num - 32) * 2, 1, 1, 8);
    }
    else{
        PairReduceSum<float>(ub_temp12, ub_temp11[4096], 1, head_num * 2, 1, 1, 8);
    }
    
    Adds<float>(ub_temp12, ub_temp12, (float)0.000001, head_num);

    // Rsqrt精度较低，使用div + sqrt提升精度
    Sqrt<float>(ub_temp12, ub_temp12, head_num);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    for (int i=0; i<head_num; i++){
        float temp = ub_temp12.GetValue(i);
        temp = ((float)1.0) / temp;
        ub_temp12.SetValue(i, temp);
    }
    
    for (int k = 0; k<head_num; k++){
        float temp = ub_temp12.GetValue(k); // scalar
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Muls<float>(ub_temp6[k*headDimQK], ub_temp6[k*headDimQK], temp, headDimQK);
    }
    Cast(src, ub_temp6, RoundMode::CAST_ODD, head_num * headDimQK);
    
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Process() {
    
    bool first_loop;
    int productSize;

    TransposeBetaHalf();
    
    PipeBarrier<PIPE_ALL>();

    TransposeGFloat();

    PipeBarrier<PIPE_ALL>();
    int32_t seq1 = 0;
    for (int bs=0; bs<batchSize; bs++){
        for (int nh=0; nh<singleCoreNumHead; nh++){
            int32_t seqLen_i = cuSeqlensGm_.GetValue(batch_i);
            if (seqLen_i <= 0) {
                continue;
            }
            if (seq1 < 0 || seq1 > static_cast<int32_t>(seqLen) || (seq1 + seqLen_i) > static_cast<int32_t>(seqLen)) {
                return;
            }
            int32_t seq0 = seq1;
            seq1 += seqLen_i;
            int head_index = nh + block_id * singleCoreNumHead;
            seqLenPadded_i = ceil_div(seqLen_i, chunkSize) * chunkSize;
            for (int nc=0; nc<(seqLenPadded_i / chunkSize); nc ++){
                if (nc == 0){
                    first_loop = true;
                }
                else{
                    first_loop = false;
                }
                
                // [SEQ_LEN, NUM_HEAD, QK_HEAD_DIM/V_HEAD_DIM] ---> [NUM_HEAD, SEQ_LEN (NUM_CHUNK, CHUNK_SIZE), QK_HEAD_DIM/V_HEAD_DIM]
                int qk_index = (seq0 * numHead + nc * numHead * chunkSize + head_index) * headDimQK;
                int v_index =  (seq0 * numHead + nc * numHead * chunkSize + head_index) * headDimV;
                int gbeta_index = seq0 * numHead + head_index * seqLenPaded + nc * chunkSize;
                int ls_index = (bs * numHead + head_index) *  headDimV * headDimQK;
                int attn_index = (bs * numHead * seqLen + nc * numHead * chunkSize + head_index) * headDimV;

                // int temp_index = (bs * numHead * seqLenPaded + head_index * seqLenPaded + nc * chunkSize) * headDimQK;
                
                LoadQKVHalfWithTail(ub_temp11.ReinterpretCast<half>(), kGlobal[qk_index], headDimQK, nc);

                if (useQKL2normInKernel){
                    // L2norm 函数使用 ub_temp6-7(32KB), ub_temp12(32KB), 以及 ub_temp11[4096](16KB) 来暂存L2norm产生的中间变量, 
                    // fp32浮点数结果会被存储在ub_temp6-7, fp16浮点数结果会进行原位替换;
                    SetFlag<HardEvent::MTE2_V>(3);
                    WaitFlag<HardEvent::MTE2_V>(3);
                    Duplicate<float>(ub_temp6, (float)0, chunkSize * headDimQK);
                    if ( (nc == ((seqLenPaded/chunkSize) - 1)) && (tailSeqLen != 0) ){ 
                        L2normDim128Float(ub_temp11.ReinterpretCast<half>(), tailSeqLen);
                    }
                    else{
                        L2normDim128Float(ub_temp11.ReinterpretCast<half>(), chunkSize);
                    }
                }
                
                SetFlag<HardEvent::MTE2_V>(1);

                DataCopy<half>(ub_temp0.ReinterpretCast<half>(), btransGlobal[gbeta_index], chunkSize);

                SetFlag<HardEvent::MTE2_V>(2);

                DataCopy<float>(ub_temp1[chunkSize], gtransGlobal[gbeta_index], chunkSize);
                SetFlag<HardEvent::MTE2_V>(0);

                WaitFlag<HardEvent::MTE2_V>(0);
                Duplicate(ub_temp1, (float)0, chunkSize);

                for (int i = 0; i < 6; i++) {
                    Add(ub_temp1[chunkSize], ub_temp1[chunkSize], ub_temp1[chunkSize - (1 << i)], chunkSize);
                }
                SetFlag<HardEvent::V_S>(0);

                WaitFlag<HardEvent::MTE2_V>(2);
                Cast(ub_temp1[chunkSize * 2], ub_temp0.ReinterpretCast<half>(), RoundMode::CAST_NONE, chunkSize);

                LocalTensor<float> g_exp_i = ub_temp1[chunkSize];    // ub_temp4-5  ub_temp1[64] 
                LocalTensor<float> beta_i = ub_temp1[chunkSize * 2]; //  ub_temp1[64-128]=g_exp_i ub_temp1[64-192]=beta_i  ub_temp4-5   ub_temp10 
                
                WaitFlag<HardEvent::MTE2_V>(1);
                // 使用fp32 vector计算代替cube计算，提升精度
                for (int i = 0; i < chunkSize; i++) {
                    float beta_temp = beta_i.GetValue(i);
                    Muls<float>(ub_temp11[headDimQK * i], ub_temp6[headDimQK * i], beta_temp, headDimQK); // ub_temp6-7=key
                    for (int j = 0; j < chunkSize; j++) {
                        int acc_index = i * chunkSize + j; 
                        Mul<float>(ub_temp8, ub_temp11[headDimQK * i], ub_temp6[headDimQK * j],  headDimQK);
                        RepeatReduceSum<float>(ub_temp8[headDimQK * 2], ub_temp8, 2, 64, 0, 1, 1, 8);
                        PairReduceSum<float>(ub_temp9[acc_index], ub_temp8[headDimQK * 2], 1, 2, 1, 1, 8);
                    }
                }

                Transpose_64_128(ub_temp4, ub_temp6, ub_temp4, ub_temp6); // ub_temp6-7=key
                LocalTensor<float> &k_trans_i = ub_temp4;
                

                WaitFlag<HardEvent::V_S>(0);
                for (int i = 0; i < chunkSize; i++) {
                    Duplicate(ub_temp2[i * chunkSize], ub_temp1[chunkSize].GetValue(i), chunkSize);
                    Sub(ub_temp2[i * chunkSize], ub_temp2[i * chunkSize], ub_temp1[chunkSize], chunkSize);
                }
                Duplicate<float>(ub_temp10, 0, chunkSize * chunkSize);
                for (int l = 0; l < chunkSize; l++) {
                    Exp<float>(ub_temp10[l * chunkSize], ub_temp2[l * chunkSize], l); // 保证对角线及以上元素为0
                }
                Exp<float>(g_exp_i, g_exp_i, chunkSize);
                SetFlag<HardEvent::V_S>(1);

                LocalTensor<float> &decay_mask_i = ub_temp10; // 差异对角线元素为0   ub_temp1[64-128]  ub_temp4-5  ub_temp10

                Mul<float>(ub_temp9, ub_temp9, decay_mask_i, chunkSize * chunkSize);
                Muls<float>(ub_temp9, ub_temp9, (float)-1, chunkSize * chunkSize);
                LocalTensor<float> &attn_tmp_i = ub_temp9;
                SetFlag<HardEvent::V_S>(0);
                for (int l = 1; l < chunkSize; l++) {
                    WaitFlag<HardEvent::V_S>(0);
                    Muls(ub_temp1[chunkSize * 3], attn_tmp_i, (attn_tmp_i[chunkSize * l].GetValue(0)), chunkSize);

                    for (int m = 1; m < l; m++) {
                        Muls(ub_temp1[chunkSize * 4], attn_tmp_i[chunkSize * m], (attn_tmp_i[chunkSize * l].GetValue(m)), chunkSize);
                        Add(ub_temp1[chunkSize * 3], ub_temp1[chunkSize * 3], ub_temp1[chunkSize * 4], chunkSize);
                    }
                    Add(attn_tmp_i[chunkSize * l], ub_temp1[chunkSize * 3], attn_tmp_i[chunkSize * l], chunkSize);
                    SetFlag<HardEvent::V_S>(0);
                }
                WaitFlag<HardEvent::V_S>(0);

                for (int l = 0; l < chunkSize; l++) {
                    attn_tmp_i[chunkSize * l].SetValue(l, (float)1.0);
                }
                SetFlag<HardEvent::S_V>(0);
                WaitFlag<HardEvent::S_V>(0);

                // for (int l = 0; l < chunkSize; l++) {
                //     Mul<float>(attn_tmp_i[chunkSize * l], attn_tmp_i[chunkSize * l], beta_i, chunkSize);
                // }
                Mul<float>(attn_tmp_i, attn_tmp_i, beta_i, chunkSize, chunkSize, {1, 1, 1, 8, 8, 0});
                // for (int l = 0; l < chunkSize; l++) {
                //     Mul<float>(ub_temp7[chunkSize * l], attn_tmp_i[chunkSize * l], g_exp_i, chunkSize);
                // }
                Mul<float>(ub_temp7, attn_tmp_i, g_exp_i, chunkSize, chunkSize, {1, 1, 1, 8, 8, 0});


                Cast(ub_temp0.ReinterpretCast<half>(), ub_temp7, RoundMode::CAST_ODD, chunkSize * chunkSize);
                Cast(ub_temp8.ReinterpretCast<half>(), k_trans_i, RoundMode::CAST_ODD, chunkSize * headDimQK);

                matmulObj_3.SetTensorA(ub_temp0.ReinterpretCast<half>());
                matmulObj_3.SetTensorB(ub_temp8.ReinterpretCast<half>(), true);
                matmulObj_3.Iterate();
                SetFlag<HardEvent::V_MTE2>(0);

                matmulObj_3.GetTensorC(ub_temp6);
                matmulObj_3.End();
                LocalTensor<float> &k_cumdecay_i = ub_temp6;

                WaitFlag<HardEvent::V_MTE2>(0);

                LoadQKVHalfWithTail(ub_temp0.ReinterpretCast<half>(), vGlobal[v_index], headDimV, nc);

                SetFlag<HardEvent::MTE2_V>(0);

                WaitFlag<HardEvent::MTE2_V>(0);

                // 使用fp32 vector计算代替cube计算，提升精度
                Cast(ub_temp12, ub_temp0.ReinterpretCast<half>(), RoundMode::CAST_NONE, chunkSize * headDimV);
                Transpose_64_128(ub_temp11, ub_temp12, ub_temp11, ub_temp12);
                for (int i = 0; i < chunkSize; i++) {
                    for (int j = 0; j < headDimV; j++) {
                        int acc_index = i * headDimV + j; 
                        Mul<float>(ub_temp12, attn_tmp_i[chunkSize * i], ub_temp11[chunkSize * j], chunkSize);
                        RepeatReduceSum<float>(ub_temp2[acc_index], ub_temp12, 1, chunkSize, 0, 1, 1, 8);
                    }
                }
                LocalTensor<float> &v_i = ub_temp2;

                // WaitFlag<HardEvent::V_MTE2>(1);
                if (first_loop) {
                    if (hasInitState) {
                        DataCopy(ub_temp11, isGlobal[ls_index], headDimV * headDimQK);
                        SetFlag<HardEvent::MTE2_V>(0);
                        WaitFlag<HardEvent::MTE2_V>(0);
                        Cast(ub_temp8.ReinterpretCast<half>(), ub_temp11, RoundMode::CAST_ODD, headDimV * headDimQK);
                    } else {
                        Duplicate(ub_temp8, (float)0, headDimV * headDimQK / 2);
                    }
                } else {
                    DataCopy(ub_temp11, lsGlobal[ls_index], headDimV * headDimQK);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    Cast(ub_temp8.ReinterpretCast<half>(), ub_temp11, RoundMode::CAST_ODD, headDimV * headDimQK);
                }
                SetFlag<HardEvent::V_MTE2>(0);

                Cast(ub_temp0.ReinterpretCast<half>(), k_cumdecay_i, RoundMode::CAST_ODD, chunkSize * headDimQK);

                matmulObj_2.SetTensorA(ub_temp0.ReinterpretCast<half>());
                matmulObj_2.SetTensorB(ub_temp8.ReinterpretCast<half>());
                matmulObj_2.IterateAll(ub_temp6);
                matmulObj_2.End();

                LocalTensor<float> &v_prime = ub_temp6;
                LocalTensor<float> &v_new = v_i;

                Sub(v_new, v_i, v_prime, chunkSize * headDimQK);
                
                // v_new存回dram保持fp32精度
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy<float>(vnewGlobal[block_id * chunkSize * headDimV], v_new, chunkSize * headDimV);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);

                LocalTensor<half> v_new_half = ub_temp0.ReinterpretCast<half>();
                Cast(v_new_half, v_new, RoundMode::CAST_ODD, chunkSize * headDimQK);

                LocalTensor<float> &q_i = ub_temp6; //ub_temp0=v_new half ub_temp1[64-128]=g_exp_i ub_temp1[64-192]=beta_i  ub_temp4-5=k_trans_i ub_temp6-7=q_i  ub_temp8-9=last_recurrent_state half ub_temp10=decay_mask_i

                WaitFlag<HardEvent::V_MTE2>(0);
                LoadQKVHalfWithTail(ub_temp11.ReinterpretCast<half>(), qGlobal[qk_index], headDimQK, nc);
                
                if (useQKL2normInKernel){
                    // L2norm 函数使用 ub_temp6-7(32KB), ub_temp12(32KB), 以及 ub_temp11[4096](16KB) 来暂存L2norm产生的中间变量, 
                    // fp32浮点数结果会被存储在ub_temp6-7, fp16浮点数结果会进行原位替换。
                    SetFlag<HardEvent::MTE2_V>(3);
                    WaitFlag<HardEvent::MTE2_V>(3);
                    Duplicate<float>(ub_temp6, (float)0, chunkSize * headDimV);
                    if ( (nc == ((seqLenPaded/chunkSize) - 1)) && (tailSeqLen != 0) ){ 
                        L2normDim128Float(ub_temp11.ReinterpretCast<half>(), tailSeqLen);
                    }
                    else{
                        L2normDim128Float(ub_temp11.ReinterpretCast<half>(), chunkSize);
                    }
                }

                float scale = ((float)1.0) / __builtin_cce_sqrtf(headDimQKfp32);
                Muls<float>(q_i, q_i, scale, chunkSize * headDimQK);

                Cast(ub_temp2.ReinterpretCast<half>(), q_i, RoundMode::CAST_ODD, chunkSize * headDimQK);
                Cast(ub_temp3.ReinterpretCast<half>(), k_trans_i, RoundMode::CAST_ODD, chunkSize * headDimQK);

                matmulObj_1.SetTensorA(ub_temp2.ReinterpretCast<half>());
                matmulObj_1.SetTensorB(ub_temp3.ReinterpretCast<half>());
                matmulObj_1.Iterate();
                matmulObj_1.GetTensorC(ub_temp2);
                matmulObj_1.End();

                for (int cs = 0; cs < chunkSize; cs++) {
                    decay_mask_i.SetValue(cs + cs * chunkSize, (float)1.0);
                }
                SetFlag<HardEvent::S_V>(2);

                WaitFlag<HardEvent::S_V>(2);
                Mul<float>(ub_temp2, ub_temp2, decay_mask_i, chunkSize * chunkSize);

                LocalTensor<float> &attn = ub_temp2; //ub_temp0=v_new half ub_temp1[64-128]=g_exp_i ub_temp1[64-192]=beta_i ub_temp2=attn  ub_temp4-5=k_trans_i ub_temp6-7=q_i  ub_temp8-9=last_recurrent_state half ub_temp10=decay_mask_i

                float g_exp_temp;
                WaitFlag<HardEvent::V_S>(1);
                for (int cs = 0; cs < chunkSize; cs++) {
                    g_exp_temp = g_exp_i.GetValue(cs);
                    Muls(q_i[cs * headDimQK], q_i[cs * headDimQK], g_exp_temp, headDimQK);
                }
                Cast(ub_temp3.ReinterpretCast<half>(), q_i, RoundMode::CAST_ODD, chunkSize * headDimQK);

                matmulObj_2.SetTensorA(ub_temp3.ReinterpretCast<half>());
                matmulObj_2.SetTensorB(ub_temp8.ReinterpretCast<half>());
                matmulObj_2.IterateAll(ub_temp6);
                matmulObj_2.End();

                LocalTensor<float> &attn_inter = ub_temp6; //ub_temp0=v_new half  ub_temp2=attn  ub_temp4-5=k_trans_i ub_temp6-7=attn_inter   ub_temp10=decay_mask_i

                Cast(ub_temp1.ReinterpretCast<half>(), attn, RoundMode::CAST_ODD, chunkSize * chunkSize);

                matmulObj_3.SetTensorA(ub_temp1.ReinterpretCast<half>());
                matmulObj_3.SetTensorB(v_new_half);

                matmulObj_3.Iterate();
                matmulObj_3.GetTensorC(ub_temp2);
                matmulObj_3.End();

                Add(attn_inter, attn_inter, ub_temp2, chunkSize * headDimQK);

                Cast(ub_temp2.ReinterpretCast<half>(), attn_inter, RoundMode::CAST_ODD, chunkSize * headDimQK);
                SetFlag<HardEvent::V_MTE3>(0);
                SetFlag<HardEvent::V_MTE2>(0);

                WaitFlag<HardEvent::V_MTE3>(0);
                
                StoreAttnHalfWithTail(attnGlobal[attn_index], ub_temp2.ReinterpretCast<half>(), headDimV, nc);
                SetFlag<HardEvent::MTE3_MTE2>(0);

                WaitFlag<HardEvent::V_MTE2>(0);
                if (first_loop) {
                    if (hasInitState) {
                        DataCopy(ub_temp6, isGlobal[ls_index], headDimV * headDimQK);
                    } else {
                        Duplicate(ub_temp6, (float)0, headDimV * headDimQK);
                    }
                    first_loop = false;
                } else {
                    DataCopy(ub_temp6, lsGlobal[ls_index], headDimV * headDimQK);
                }
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Muls<float>(ub_temp6, ub_temp6, g_exp_temp, headDimV * headDimQK);

                Mul<float>(k_trans_i, k_trans_i, decay_mask_i[(chunkSize - 1) * chunkSize], chunkSize, headDimQK, {1, 1, 1, 8, 8, 0});

                // 使用fp32 vector计算代替cube计算，提升精度
                WaitFlag<HardEvent::MTE3_MTE2>(0);
                DataCopy<float>(ub_temp2, vnewGlobal[block_id * chunkSize * headDimV], chunkSize * headDimV);
                SetFlag<HardEvent::MTE2_V>(0);

                WaitFlag<HardEvent::MTE2_V>(0);
                Transpose_64_128(ub_temp10, ub_temp2, ub_temp10, ub_temp2);

                if ( (nc == ((seqLenPaded/chunkSize) - 1)) && (tailSeqLen != 0) ){
                    productSize = tailSeqLen;
                }
                else{
                    productSize = chunkSize;
                }
                SetFlag<HardEvent::S_V>(0);

                WaitFlag<HardEvent::S_V>(0);
                for (int i = 0; i < headDimQK; i++) {
                    for (int j = 0; j < headDimV; j++) {
                        int acc_index = i * headDimV + j; 
                        Mul<float>(ub_temp12, k_trans_i[chunkSize * i], ub_temp10[chunkSize * j], productSize);
                        RepeatReduceSum<float>(ub_temp0[acc_index], ub_temp12, 1, productSize, 0, 1, 1, 8);
                        
                    }
                }


                Add(ub_temp6, ub_temp6, ub_temp0, headDimV * headDimQK);
                SetFlag<HardEvent::V_MTE3>(0);


                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy<float>(lsGlobal[ls_index], ub_temp6, headDimV * headDimQK);
                
            }
            PipeBarrier<PIPE_ALL>();
        }
        
    }

}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Transpose_64_128(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> tmp1, LocalTensor<float> tmp2) {
    DataCopyParams repeatParams;
    repeatParams.blockLen = 1;
    repeatParams.srcGap = 15;
    repeatParams.dstGap = 0;
    repeatParams.blockCount = 64;
    PipeBarrier<PIPE_ALL>();

    for (int i = 0; i < 16; i++) {
        DataCopy<float>(tmp1[64 * 8 * i], src[8 * i], repeatParams);
    }

    PipeBarrier<PIPE_ALL>();
    uint64_t inputPtr[16];
    uint64_t outputPtr[16];
    inputPtr[0] = reinterpret_cast<uint64_t>(tmp1.GetPhyAddr());
    for (int i = 1; i < 16; i++) {
        inputPtr[i] = inputPtr[i - 1] + 32;
    }
    outputPtr[0] = reinterpret_cast<uint64_t>(tmp2.GetPhyAddr());
    outputPtr[1] = outputPtr[0] + 32;
    for (int i = 2; i < 16; i++) {
        outputPtr[i] = outputPtr[i - 2] + 128 * 64 / 8 * 4;
    }
    TransDataTo5HDParams nchwconvParams;
    nchwconvParams.repeatTimes = 128 * 64 / 16 / 8;
    nchwconvParams.dstRepStride = 2;
    nchwconvParams.srcRepStride = 16;

    PipeBarrier<PIPE_ALL>();
    TransDataTo5HD<float>(outputPtr, inputPtr, nchwconvParams);

    PipeBarrier<PIPE_ALL>();

    repeatParams.blockLen = 8;
    repeatParams.srcGap = 120;
    repeatParams.dstGap = 0;
    repeatParams.blockCount = 8;
    // DataCopy(ubSec4, ubSec3, repeatParams);
    for (int i = 0; i < 16; i++) {
        DataCopy<float>(dst[64 * 8 * i], tmp2[64 * i], repeatParams);
    }
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::LoadAndCast(LocalTensor<float> dst, GlobalTensor<half> src, uint32_t size, LocalTensor<half> tmp) {
    DataCopy<half>(tmp, src, size);
    PipeBarrier<PIPE_ALL>();
    Cast(dst, tmp, RoundMode::CAST_NONE, size);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::tmp() {
    LocalTensor<half> v_new_half = ub_temp0.ReinterpretCast<half>();
    matmulObj_3.SetTensorA(ub_temp1.ReinterpretCast<half>());
    matmulObj_3.SetTensorB(v_new_half);

    while (matmulObj_3.template Iterate<true>()) { // Once Iterate, compute baseM * baseN, sync is set true here.
        matmulObj_3.template GetTensorC<true>(ub_temp2, false, true);
        break;
    }
    matmulObj_3.End();
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::TransposeGFloat()
{
    int blockLen = 16;
    int width = 8;
    int headGroupNum = numHead / width;
    int seqLenGroupNum = seqLenPaded / blockLen;
    int blockTailSeqLen = seqLen % blockLen;
    int fullBlockGroupNum = ceil_div(seqLen, blockLen);
    
    DataCopyParams repeatParams_in;
    repeatParams_in.blockLen = width * 4 / 32;
    repeatParams_in.srcGap = width * (headGroupNum - 1) * 4 / 32;
    repeatParams_in.dstGap = 0;
    repeatParams_in.blockCount = blockLen;

    DataCopyParams repeatParams_in_tail;
    repeatParams_in_tail.blockLen = width * 4 / 32;
    repeatParams_in_tail.srcGap = width * (headGroupNum - 1) * 4 / 32;
    repeatParams_in_tail.dstGap = 0;
    repeatParams_in_tail.blockCount = blockTailSeqLen;

    DataCopyParams  repeatParams_out;
    repeatParams_out.blockLen = blockLen * 4 / 32;
    repeatParams_out.srcGap = 0;
    repeatParams_out.dstGap = blockLen * (seqLenGroupNum - 1) * 4 / 32;
    repeatParams_out.blockCount = width;

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = headGroupNum * seqLenGroupNum;
    transDataParams.dstRepStride = blockLen;
    transDataParams.srcRepStride = blockLen;

    uint64_t dstLocalList[16];
    for (int b = 0; b < 16; b++) {
        dstLocalList[b] = (uint64_t)(ub_temp7[width * b].GetPhyAddr());
    }
    uint64_t srcLocalList[16];
    for (int b = 0; b < 16; b++) {
        srcLocalList[b] = (uint64_t)(ub_temp0[width * b].GetPhyAddr());
    }

    for (int i = 0; i < batchSize; i++){
        ///////////////////////////////  g //////////////////////////////
        for(int r=0; r < (headGroupNum*seqLenGroupNum); r++){
            int j_index = r / seqLenGroupNum;
            int k_index = r % seqLenGroupNum;
            if (k_index < fullBlockGroupNum){
                int src_index = i * numHead * seqLen + k_index * numHead * blockLen + j_index * width;
                if (k_index == (fullBlockGroupNum - 1)){
                    // 最后一个块分类执行, 如果余数不为0, 则搬移剩下的块; 余数为0, 说明可以整除, 则搬运全部数据
                    if (blockTailSeqLen != 0){
                        Duplicate(ub_temp0[r*width*blockLen], (float)0, width*blockLen); // vector
                        SetFlag<HardEvent::V_MTE2>(0);
                        WaitFlag<HardEvent::V_MTE2>(0);
                        DataCopy<float>(ub_temp0[r*width*blockLen], gGlobal[src_index], repeatParams_in_tail);
                    }
                    else{
                        DataCopy<float>(ub_temp0[r*width*blockLen], gGlobal[src_index], repeatParams_in);
                    }
                }
                else{
                    // 其他块正常搬移
                    DataCopy<float>(ub_temp0[r*width*blockLen], gGlobal[src_index], repeatParams_in);
                }
            }
            else{
                // padding块搬移全0
                Duplicate(ub_temp0[r*width*blockLen], (float)0, width*blockLen);
            }
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        
        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParams);
        
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        for(int r=0; r < (headGroupNum*seqLenGroupNum); r++){
            int j_index = r / seqLenGroupNum;
            int k_index = r % seqLenGroupNum;
            int dst_index = i * numHead * seqLenPaded  + j_index * seqLenPaded * width + k_index * blockLen;
            DataCopy<float>(gtransGlobal[dst_index], ub_temp7[r*width*blockLen], repeatParams_out);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::TransposeBetaHalf()
{
    int blockLen = 16;
    int headGroupNum = numHead / blockLen;
    int seqLenGroupNum = seqLenPaded / blockLen;
    int blockTailSeqLen = seqLen % blockLen;
    int fullBlockGroupNum = ceil_div(seqLen, blockLen);
    
    DataCopyParams repeatParams_in;
    repeatParams_in.blockLen = blockLen * 2 / 32;
    repeatParams_in.srcGap = blockLen * (headGroupNum - 1) * 2 / 32;
    repeatParams_in.dstGap = 0;
    repeatParams_in.blockCount = blockLen;

    DataCopyParams repeatParams_in_tail;
    repeatParams_in_tail.blockLen = blockLen * 2 / 32;
    repeatParams_in_tail.srcGap = blockLen * (headGroupNum - 1) * 2 / 32;
    repeatParams_in_tail.dstGap = 0;
    repeatParams_in_tail.blockCount = blockTailSeqLen;

    DataCopyParams  repeatParams_out;
    repeatParams_out.blockLen = blockLen * 2 / 32;
    repeatParams_out.srcGap = 0;
    repeatParams_out.dstGap = blockLen * (seqLenGroupNum - 1) * 2 / 32;
    repeatParams_out.blockCount = blockLen;

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = headGroupNum * seqLenGroupNum;
    transDataParams.dstRepStride = blockLen;
    transDataParams.srcRepStride = blockLen;

    LocalTensor<half> ub_temp_half_1 = ub_temp0.ReinterpretCast<half>();
    LocalTensor<half> ub_temp_half_2 = ub_temp7.ReinterpretCast<half>();

    uint64_t dstLocalList[16];
    for (int b = 0; b < 16; b++) {
        dstLocalList[b] = (uint64_t)(ub_temp_half_2[blockLen * b].GetPhyAddr());
    }
    uint64_t srcLocalList[16];
    for (int b = 0; b < 16; b++) {
        srcLocalList[b] = (uint64_t)(ub_temp_half_1[blockLen * b].GetPhyAddr());
    }

    for (int i = 0; i < batchSize; i++){
        ///////////////////////////////  beta //////////////////////////////
        for(int r=0; r < (headGroupNum*seqLenGroupNum); r++){
            int j_index = r / seqLenGroupNum;
            int k_index = r % seqLenGroupNum;
            if (k_index < fullBlockGroupNum){
                int src_index = i * numHead * seqLen + k_index * numHead * blockLen + j_index * blockLen;
                if (k_index == (fullBlockGroupNum - 1)){
                    // 最后一个块分类执行, 如果余数不为0, 则搬移剩下的块; 余数为0, 说明可以整除, 则搬运全部数据
                    if (blockTailSeqLen != 0){
                        Duplicate(ub_temp_half_1[r*blockLen*blockLen], (half)0, blockLen*blockLen);
                        SetFlag<HardEvent::V_MTE2>(0);
                        WaitFlag<HardEvent::V_MTE2>(0);
                        DataCopy<half>(ub_temp_half_1[r*blockLen*blockLen], bGlobal[src_index], repeatParams_in_tail);
                    }
                    else{
                        DataCopy<half>(ub_temp_half_1[r*blockLen*blockLen], bGlobal[src_index], repeatParams_in);
                    }
                }
                else{
                    // 其他块正常搬移
                    DataCopy<half>(ub_temp_half_1[r*blockLen*blockLen], bGlobal[src_index], repeatParams_in);
                }
                
            }
            else{
                // padding块搬移全0
                Duplicate(ub_temp_half_1[r*blockLen*blockLen], (half)0, blockLen*blockLen);
            }
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
        
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        for(int r=0; r < (headGroupNum*seqLenGroupNum); r++){
            int j_index = r / seqLenGroupNum;
            int k_index = r % seqLenGroupNum;
            int dst_index = i * numHead * seqLenPaded  + j_index * seqLenPaded * blockLen + k_index * blockLen;
            DataCopy<half>(btransGlobal[dst_index], ub_temp_half_2[r*blockLen*blockLen], repeatParams_out);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_v310(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR initial_state, GM_ADDR actual_seq_lengths, GM_ADDR core_attn, GM_ADDR last_recurrent_state, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    ChunkGatedDeltaRuleV310Kernel kernel;
    kernel.Init(query, key, value, g, beta, initial_state, actual_seq_lengths, core_attn, last_recurrent_state, workspace, tiling_data, &pipe);
    kernel.Process();
    PipeBarrier<PIPE_ALL>();
}

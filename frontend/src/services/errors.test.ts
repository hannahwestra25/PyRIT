import { toApiError } from './errors'
import type { ApiError } from './errors'

// ---------------------------------------------------------------------------
// Helper: build a fake AxiosError-like object
// ---------------------------------------------------------------------------
function makeAxiosError(opts: {
  status?: number
  data?: unknown
  code?: string
  message?: string
  hasResponse?: boolean
}): unknown {
  const { status, data, code, message = 'Request failed', hasResponse = true } = opts

  const err: Record<string, unknown> = {
    isAxiosError: true,
    message,
    code,
    config: {},
    toJSON: () => ({}),
  }

  if (hasResponse && status !== undefined) {
    err.response = { status, data, headers: {}, config: {} }
  }

  return err
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('toApiError', () => {
  // 1. Axios error with RFC 7807 JSON body
  it('extracts detail and type from RFC 7807 JSON body', () => {
    const err = makeAxiosError({
      status: 400,
      data: {
        type: 'https://tools.ietf.org/html/rfc7807',
        title: 'Bad Request',
        status: 400,
        detail: 'Conversation ID is required.',
      },
    })

    const result: ApiError = toApiError(err)

    expect(result.status).toBe(400)
    expect(result.detail).toBe('Conversation ID is required.')
    expect(result.type).toBe('https://tools.ietf.org/html/rfc7807')
    expect(result.isNetworkError).toBe(false)
    expect(result.isTimeout).toBe(false)
    expect(result.raw).toBe(err)
  })

  // 2. Axios error with plain-string body (e.g. proxy HTML)
  it('sanitizes a plain-string server error response', () => {
    const internalDetail = '<html>C:\\internal\\proxy.conf secret=sk-test</html>'
    const err = makeAxiosError({
      status: 502,
      data: internalDetail,
    })

    const result = toApiError(err)

    expect(result.status).toBe(502)
    expect(result.detail).toBe('The server could not complete the request. Please try again.')
    expect(result.detail).not.toContain(internalDetail)
    expect(result.type).toBeUndefined()
    expect(result.isNetworkError).toBe(false)
  })

  // 3. Axios error with HTML body (nginx 502 — object without detail)
  it('falls back to status text when JSON body lacks detail', () => {
    const err = makeAxiosError({
      status: 500,
      data: { error: 'internal' }, // no 'detail' field
    })

    const result = toApiError(err)

    expect(result.status).toBe(500)
    expect(result.detail).toBe('The server could not complete the request. Please try again.')
    expect(result.type).toBeUndefined()
  })

  it('preserves an explicitly categorized server error', () => {
    const err = makeAxiosError({
      status: 503,
      data: {
        type: '/errors/target-unavailable',
        title: 'Target Unavailable',
        status: 503,
        detail: 'The target is temporarily unavailable. Please try again later.',
      },
    })

    const result = toApiError(err)

    expect(result.status).toBe(503)
    expect(result.detail).toBe('The target is temporarily unavailable. Please try again later.')
    expect(result.type).toBe('/errors/target-unavailable')
  })

  it('sanitizes an untyped server error detail', () => {
    const internalDetail = 'provider secret=sk-test failed at C:\\internal\\provider.py'
    const err = makeAxiosError({
      status: 500,
      data: { detail: internalDetail },
    })

    const result = toApiError(err)

    expect(result.detail).toBe('The server could not complete the request. Please try again.')
    expect(result.detail).not.toContain(internalDetail)
  })

  it('sanitizes an internal RFC 7807 server error detail', () => {
    const internalDetail = 'provider secret=sk-test failed at C:\\internal\\provider.py'
    const err = makeAxiosError({
      status: 500,
      data: {
        type: '/errors/internal-error',
        title: 'Internal Server Error',
        status: 500,
        detail: internalDetail,
      },
    })

    const result = toApiError(err)

    expect(result.detail).toBe('The server could not complete the request. Please try again.')
    expect(result.detail).not.toContain(internalDetail)
    expect(result.type).toBeUndefined()
  })

  it('sanitizes an unreviewed RFC 7807 server error type', () => {
    const internalDetail = 'database secret=sk-test failed at C:\\internal\\database.py'
    const err = makeAxiosError({
      status: 503,
      data: {
        type: '/errors/database-error',
        title: 'Database Error',
        status: 503,
        detail: internalDetail,
      },
    })

    const result = toApiError(err)

    expect(result.detail).toBe('The server could not complete the request. Please try again.')
    expect(result.detail).not.toContain(internalDetail)
  })

  it.each([
    {
      name: 'a mismatched status',
      data: {
        type: '/errors/target-unavailable',
        title: 'Target Unavailable',
        status: 200,
        detail: 'provider secret=sk-test',
      },
    },
    {
      name: 'a missing title',
      data: {
        type: '/errors/target-unavailable',
        status: 503,
        detail: 'provider secret=sk-test',
      },
    },
  ])('sanitizes a server problem with $name', ({ data }) => {
    const result = toApiError(makeAxiosError({ status: 503, data }))

    expect(result.detail).toBe('The server could not complete the request. Please try again.')
    expect(result.detail).not.toContain('sk-test')
  })

  // 4. Axios error with no response (network / CORS)
  it('returns isNetworkError for Axios errors without response', () => {
    const err = makeAxiosError({
      hasResponse: false,
      message: 'Network Error',
    })

    const result = toApiError(err)

    expect(result.status).toBeNull()
    expect(result.detail).toBe('Network error — check that the backend is running and reachable.')
    expect(result.isNetworkError).toBe(true)
    expect(result.isTimeout).toBe(false)
  })

  // 5. Axios timeout error
  it('returns isTimeout for ECONNABORTED', () => {
    const err = makeAxiosError({
      hasResponse: false,
      code: 'ECONNABORTED',
      message: 'timeout of 5000ms exceeded',
    })

    const result = toApiError(err)

    expect(result.status).toBeNull()
    expect(result.detail).toContain('timed out')
    expect(result.isTimeout).toBe(true)
    expect(result.isNetworkError).toBe(false)
  })

  // 6. Non-Axios Error instance
  it('sanitizes non-Axios Error messages', () => {
    const err = new Error('secret=sk-test at C:\\internal\\runtime.ts')

    const result = toApiError(err)

    expect(result.status).toBeNull()
    expect(result.detail).toBe('An unexpected error occurred.')
    expect(result.detail).not.toContain('sk-test')
    expect(result.isNetworkError).toBe(false)
    expect(result.isTimeout).toBe(false)
    expect(result.raw).toBe(err)
  })

  // 7. String throw
  it('sanitizes string throws', () => {
    const result = toApiError('provider deployment secret-model failed')

    expect(result.status).toBeNull()
    expect(result.detail).toBe('An unexpected error occurred.')
    expect(result.detail).not.toContain('secret-model')
    expect(result.isNetworkError).toBe(false)
  })

  // 8. Unknown throw (null)
  it('returns generic message for null throw', () => {
    const result = toApiError(null)

    expect(result.detail).toBe('An unexpected error occurred.')
    expect(result.raw).toBeNull()
  })

  // 9. Unknown throw (undefined)
  it('returns generic message for undefined throw', () => {
    const result = toApiError(undefined)

    expect(result.detail).toBe('An unexpected error occurred.')
    expect(result.raw).toBeUndefined()
  })

  // 10. Axios error with null data
  it('falls back to status text when response data is null', () => {
    const err = makeAxiosError({ status: 403, data: null })

    const result = toApiError(err)

    expect(result.status).toBe(403)
    expect(result.detail).toBe('Server error (403)')
  })

  it('does not expose plain-string client error bodies', () => {
    const internalDetail = 'provider token secret=sk-test at C:\\internal\\proxy.conf'
    const err = makeAxiosError({ status: 400, data: internalDetail })

    const result = toApiError(err)

    expect(result.detail).toBe('Server error (400)')
    expect(result.detail).not.toContain(internalDetail)
  })

  // 11. Error with empty message
  it('falls back to generic message for Error with empty message', () => {
    const err = new Error('')

    const result = toApiError(err)

    expect(result.detail).toBe('An unexpected error occurred.')
  })

  // 12. Axios 422 with RFC 7807 (validation error)
  it('handles 422 validation errors with detail field', () => {
    const err = makeAxiosError({
      status: 422,
      data: {
        type: 'validation_error',
        detail: 'Field "name" is required.',
        errors: [{ field: 'name', message: 'required' }],
      },
    })

    const result = toApiError(err)

    expect(result.status).toBe(422)
    expect(result.detail).toBe('Field "name" is required.')
    expect(result.type).toBe('validation_error')
  })
})
